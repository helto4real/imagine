use std::io::{self, BufRead, IsTerminal, Write};
fn main() -> io::Result<()> {
    let stdout = io::stdout();
    let color = st_proxy::logging::colors_enabled(
        stdout.is_terminal(),
        std::env::var("NO_COLOR").ok().as_deref(),
        std::env::var("TERM").ok().as_deref(),
    );
    let mut out = stdout.lock();
    let mut input = io::stdin().lock();
    let mut line = Vec::new();
    while input.read_until(b'\n', &mut line)? > 0 {
        if color {
            let mut text = String::new();
            let mut rest = line.as_slice();
            while !rest.is_empty() {
                match std::str::from_utf8(rest) {
                    Ok(s) => {
                        text.push_str(&s.replace('\u{f0000}', "\u{f0000}\u{f0000}"));
                        break;
                    }
                    Err(e) => {
                        let n = e.valid_up_to();
                        text.push_str(
                            &std::str::from_utf8(&rest[..n])
                                .unwrap()
                                .replace('\u{f0000}', "\u{f0000}\u{f0000}"),
                        );
                        let bad = e.error_len().unwrap_or(rest.len() - n);
                        for b in &rest[n..n + bad] {
                            text.push('\u{f0000}');
                            text.push(char::from_u32(0xf0100 + u32::from(*b)).unwrap());
                        }
                        rest = &rest[n + bad..];
                    }
                }
            }
            let text = st_proxy::logging::strata_line(&text);
            let mut chars = text.chars();
            while let Some(c) = chars.next() {
                if c == '\u{f0000}' {
                    let next = chars.next().expect("escaped marker");
                    if next == c {
                        out.write_all(c.encode_utf8(&mut [0; 4]).as_bytes())?;
                    } else {
                        out.write_all(&[(next as u32 - 0xf0100) as u8])?;
                    }
                } else {
                    out.write_all(c.encode_utf8(&mut [0; 4]).as_bytes())?;
                }
            }
        } else {
            out.write_all(&line)?
        }
        out.flush()?;
        line.clear();
    }
    Ok(())
}
