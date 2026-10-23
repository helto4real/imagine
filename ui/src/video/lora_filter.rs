//! Optional ECMAScript catalog filtering, not LoRA admission or inference.
//!
//! Nonempty filters use installed `node` (PATH) as an optional runtime dependency.
//! No Rust regex approximation is substituted when it is missing. The owned helper
//! receives only in-memory patterns/filenames over pipes and never loads a model.
use std::{
    ffi::OsStr,
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
};

// Director web/timeline/lora_editor.js::loraEditorFilteredChoices:
// new RegExp(needle), no flags; invalid expressions return every nonempty name.
const SCRIPT: &str = r#"
const rl = require('node:readline').createInterface({input:process.stdin,crlfDelay:Infinity});
rl.on('line', line => {
  let request;
  try {
    request = JSON.parse(line);
    const source = request.files.map((file,index) => ({file:String(file ?? ''),index})).filter(item => item.file);
    let choices = source;
    let invalid = false;
    try {
      const needle = String(request.pattern ?? '');
      if (needle) {
        const regex = new RegExp(needle);
        choices = source.filter(item => regex.test(item.file));
      }
    } catch { invalid = true; choices = source; }
    process.stdout.write(JSON.stringify({id:request.id,indices:choices.map(item=>item.index),invalid})+'\n');
  } catch { process.stdout.write(JSON.stringify({id:request?.id,error:true})+'\n'); }
});
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Error {
    Unavailable,
    Failed,
}
impl Error {
    pub fn message(self) -> &'static str {
        match self {
            Self::Unavailable => {
                "ECMAScript filter unavailable: optional Node.js could not start. Showing all files; manual paths and Apply remain available."
            }
            Self::Failed => {
                "ECMAScript filter failed. Showing all files; manual paths and Apply remain available. Refresh to retry."
            }
        }
    }
}
pub(super) struct Matches {
    pub indices: Vec<usize>,
    pub invalid: bool,
}
pub(super) struct Reply {
    pub nonce: u64,
    pub result: Result<Matches, Error>,
}
struct Request {
    nonce: u64,
    files: Vec<String>,
    pattern: String,
    context: eframe::egui::Context,
}
/// One editor-owned helper; Drop kills/reaps this Child only, then joins its pipe worker.
pub(super) struct Engine {
    child: Arc<Mutex<Child>>,
    requests: Option<Sender<Request>>,
    replies: Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
}
fn stop(child: &Mutex<Child>) {
    let mut child = child.lock().unwrap_or_else(|e| e.into_inner());
    if !matches!(child.try_wait(), Ok(Some(_))) {
        let _ = child.kill();
    }
    let _ = child.wait();
}
impl Engine {
    pub fn start(program: &OsStr) -> Result<Self, Error> {
        let mut child = Command::new(program)
            .args(["--input-type=commonjs", "--eval", SCRIPT])
            // A controlled pipe helper must not execute optional user preload code.
            .env_remove("NODE_OPTIONS")
            .env_remove("NODE_PATH")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| Error::Unavailable)?;
        let input = child.stdin.take().expect("piped input");
        let output = child.stdout.take().expect("piped output");
        let child = Arc::new(Mutex::new(child));
        let cleanup = child.clone();
        let (tx, rx) = mpsc::channel::<Request>();
        let (reply_tx, replies) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("lora-ecmascript-filter".into())
            .spawn(move || {
                let mut input = input;
                let mut output = BufReader::new(output);
                while let Ok(request) = rx.recv() {
                    let result = evaluate(&mut input, &mut output, &request);
                    let failed = result.is_err();
                    let _ = reply_tx.send(Reply {
                        nonce: request.nonce,
                        result,
                    });
                    request.context.request_repaint();
                    if failed {
                        break;
                    }
                }
                stop(&cleanup);
            })
            .map_err(|_| {
                stop(&child);
                Error::Unavailable
            })?;
        Ok(Self {
            child,
            requests: Some(tx),
            replies,
            thread: Some(thread),
        })
    }
    pub fn submit(
        &self,
        nonce: u64,
        files: Vec<String>,
        pattern: String,
        context: &eframe::egui::Context,
    ) -> Result<(), Error> {
        self.requests
            .as_ref()
            .ok_or(Error::Failed)?
            .send(Request {
                nonce,
                files,
                pattern,
                context: context.clone(),
            })
            .map_err(|_| Error::Failed)
    }
    pub fn try_reply(&self) -> Option<Reply> {
        self.replies.try_recv().ok()
    }
    #[cfg(test)]
    pub fn child_id(&self) -> u32 {
        self.child.lock().unwrap().id()
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.requests.take();
        stop(&self.child);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
fn evaluate(
    input: &mut impl Write,
    output: &mut impl BufRead,
    request: &Request,
) -> Result<Matches, Error> {
    let id = request.nonce.to_string();
    serde_json::to_writer(
        &mut *input,
        &serde_json::json!({"id":id,"files":request.files,"pattern":request.pattern}),
    )
    .map_err(|_| Error::Failed)?;
    input
        .write_all(b"\n")
        .and_then(|_| input.flush())
        .map_err(|_| Error::Failed)?;
    let mut line = String::new();
    if output.read_line(&mut line).map_err(|_| Error::Failed)? == 0 {
        return Err(Error::Failed);
    }
    #[derive(serde::Deserialize)]
    struct Wire {
        id: String,
        indices: Vec<usize>,
        invalid: bool,
    }
    let response: Wire = serde_json::from_str(&line).map_err(|_| Error::Failed)?;
    if response.id != id
        || response.indices.iter().any(|i| *i >= request.files.len())
        || response.indices.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::Failed);
    }
    Ok(Matches {
        indices: response.indices,
        invalid: response.invalid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wait(engine: &Engine) -> Reply {
        let start = std::time::Instant::now();
        loop {
            if let Some(reply) = engine.try_reply() {
                return reply;
            }
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "synthetic engine fixture did not answer"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    #[test]
    fn node_matches_director_new_regexp_without_flags_and_invalid_fallback() {
        let engine = Engine::start(OsStr::new("node")).expect("installed Node test dependency");
        let context = eframe::egui::Context::default();
        let files: Vec<_> = [
            "people/Face.weights",
            "style/face.weights",
            "style/ink.weights",
            "same-same",
            "1",
            "١",
            "é",
            "😀",
            "",
        ]
        .map(String::from)
        .into();
        // Expected results follow Director's actual new RegExp(pattern) contract,
        // including classic (non-u) UTF-16 and ASCII character-class semantics.
        for (nonce, (pattern, expected, invalid)) in [
            ("face", vec![1], false),
            ("(?i)face", (0..8).collect(), true),
            ("[", (0..8).collect(), true),
            ("style/(?=ink)", vec![2], false),
            (r"^(\w+)-\1$", vec![3], false),
            (r"^\d$", vec![4], false),
            (r"^\w$", vec![4], false),
            (r"^.$", vec![4, 5, 6], false),
            (r"^\uD83D\uDE00$", vec![7], false),
            ("missing", vec![], false),
            ("", (0..8).collect(), false),
        ]
        .into_iter()
        .enumerate()
        {
            engine
                .submit(nonce as u64, files.clone(), pattern.into(), &context)
                .unwrap();
            let reply = wait(&engine);
            assert_eq!(reply.nonce, nonce as u64);
            let result = reply.result.unwrap();
            assert_eq!(result.indices, expected, "{pattern}");
            assert_eq!(result.invalid, invalid, "{pattern}");
        }
    }
    #[test]
    fn batch_results_match_actual_director_filter_source() {
        // Explicit approved source-only oracle, never a connected ComfyUI runtime.
        let source = std::fs::read_to_string(
            "/home/thhel/git/comfyui-helto-director/web/timeline/lora_editor.js",
        )
        .unwrap();
        let start = source
            .find("export function loraEditorFilteredChoices(")
            .unwrap();
        let end = start
            + source[start..]
                .find("\nexport async function fetchTimelineLoras(")
                .unwrap();
        let function = source[start..end].replacen("export function", "function", 1);
        let files = vec![
            "own/Face", "own/face", "ab-ab", "1", "١", "é", "😀", "", "own/face",
        ];
        let patterns = vec![
            "face",
            "(?i)face",
            "[",
            r"(?<=own/)face",
            r"^(\w+)-\1$",
            r"^\d$",
            r"^\w$",
            r"^.$",
            r"^\uD83D\uDE00$",
            r"\q",
            "",
        ];
        let script = "let raw='';process.stdin.setEncoding('utf8');process.stdin.on('data',s=>raw+=s);process.stdin.on('end',()=>{const r=JSON.parse(raw);const f=require('node:vm').runInNewContext('('+r.function+')');process.stdout.write(JSON.stringify(r.patterns.map(p=>f(r.files,p))))});";
        let mut oracle = Command::new("node")
            .args(["--input-type=commonjs", "--eval", script])
            .env_remove("NODE_OPTIONS")
            .env_remove("NODE_PATH")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut input = oracle.stdin.take().unwrap();
        serde_json::to_writer(
            &mut input,
            &serde_json::json!({"function":function,"files":files,"patterns":patterns}),
        )
        .unwrap();
        drop(input);
        let output = oracle.wait_with_output().unwrap();
        assert!(output.status.success());
        let expected: Vec<Vec<String>> = serde_json::from_slice(&output.stdout).unwrap();
        let engine = Engine::start(OsStr::new("node")).unwrap();
        for (index, pattern) in patterns.into_iter().enumerate() {
            engine
                .submit(
                    index as u64,
                    files.iter().map(|s| (*s).into()).collect(),
                    pattern.into(),
                    &eframe::egui::Context::default(),
                )
                .unwrap();
            let result = wait(&engine).result.unwrap();
            let actual: Vec<_> = result.indices.iter().map(|i| files[*i]).collect();
            assert_eq!(actual, expected[index], "{pattern}");
        }
    }
    #[test]
    fn helper_lifecycle_reaps_only_its_child_and_missing_engine_is_nonfatal() {
        assert!(matches!(
            Engine::start(OsStr::new("/own/nonexistent-node-engine")),
            Err(Error::Unavailable)
        ));
        let engine = Engine::start(OsStr::new("node")).unwrap();
        let child = engine.child.clone();
        assert!(child.lock().unwrap().try_wait().unwrap().is_none());
        drop(engine);
        assert!(child.lock().unwrap().try_wait().unwrap().is_some());
    }
}
