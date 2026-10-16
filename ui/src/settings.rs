use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

pub const DEFAULT_API_URL: &str = "http://127.0.0.1:8189/imagine";

#[derive(Clone)]
pub struct Connection {
    pub url: String,
    pub access: PathBuf,
    pub client: String,
}
impl Connection {
    pub fn token(&self) -> Result<String> {
        #[derive(Deserialize)]
        struct Access {
            clients: BTreeMap<String, String>,
        }
        let access: Access = serde_json::from_slice(
            &std::fs::read(&self.access)
                .context("Start the server with ./start-server.sh, then select Reconnect.")?,
        )
        .context("Could not read the API key. Check access.json.")?;
        access
            .clients
            .get(&self.client)
            .cloned()
            .context("The client is missing from access.json.")
    }
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.url).context("Invalid API URL.")?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            bail!(
                "The API URL must use HTTP or HTTPS without credentials, query parameters or a fragment."
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Preferences {
    pub output_folder: PathBuf,
    #[serde(default = "privacy_default")]
    pub privacy_mode: bool,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default)]
    pub generation: GenerationSettings,
    #[serde(default = "sheet_width")]
    pub sheet_width: u32,
    #[serde(default)]
    pub upscale_target: UpscaleTarget,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpscaleTarget {
    #[default]
    #[serde(rename = "2k")]
    K2,
    #[serde(rename = "3k")]
    K3,
    #[serde(rename = "4k")]
    K4,
}
impl UpscaleTarget {
    pub const ALL: [Self; 3] = [Self::K2, Self::K3, Self::K4];
    pub fn label(self) -> &'static str {
        match self {
            Self::K2 => "2K",
            Self::K3 => "3K",
            Self::K4 => "4K",
        }
    }
    pub fn max_side(self) -> u32 {
        match self {
            Self::K2 => 2048,
            Self::K3 => 3072,
            Self::K4 => 4096,
        }
    }
    pub fn dimensions(self, ratio: &str) -> Result<(u32, u32)> {
        GenerationSettings {
            max_side: self.max_side(),
            aspect_ratio: ratio.into(),
            ..Default::default()
        }
        .dimensions()
    }
}
// AIO services/dimensions.py, preset ratio mode with multiple value 32.
pub const ASPECT_RATIOS: [&str; 9] = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationSettings {
    pub steps: u32,
    pub cfg_scale: f32,
    pub random_seed: bool,
    pub seed: u32,
    pub aspect_ratio: String,
    pub max_side: u32,
    pub reference_max_pixels_per_image: u32,
}
impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            steps: 40,
            cfg_scale: 2.5,
            random_seed: true,
            seed: 42,
            aspect_ratio: "16:9".into(),
            max_side: 1600,
            reference_max_pixels_per_image: 1_000_000,
        }
    }
}
impl GenerationSettings {
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        if !(256..=4096).contains(&self.max_side) || !self.max_side.is_multiple_of(32) {
            bail!("Max side must be 256–4096 px and a multiple of 32.");
        }
        if !ASPECT_RATIOS.contains(&self.aspect_ratio.as_str()) {
            bail!("Select a valid aspect ratio.");
        }
        let (w, h) = self.aspect_ratio.split_once(':').unwrap();
        let w: u32 = w.parse()?;
        let h: u32 = h.parse()?;
        // Python round uses ties-to-even, including 16:9 at max side 256.
        let short = ((self.max_side as f64 * w.min(h) as f64 / w.max(h) as f64 / 32.)
            .round_ties_even() as u32
            * 32)
            .max(32);
        Ok(if w >= h {
            (self.max_side, short)
        } else {
            (short, self.max_side)
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.dimensions()?;
        if !(262_144..=4_000_000).contains(&self.reference_max_pixels_per_image) {
            bail!("Reference size must be 0.262144–4 MP per image.");
        }
        if !(1..=100).contains(&self.steps)
            || !self.cfg_scale.is_finite()
            || !(0.0..=20.0).contains(&self.cfg_scale)
        {
            bail!("Steps must be 1–100 and CFG 0–20.");
        }
        Ok(())
    }
}
fn sheet_width() -> u32 {
    1536
}
fn privacy_default() -> bool {
    true
}
impl Preferences {
    pub fn load(project: &Path) -> Result<Self> {
        let path = project.join(".imagine/ui-settings.json");
        if !path.exists() {
            return Ok(Self {
                output_folder: project.join(".imagine/images"),
                privacy_mode: true,
                negative_prompt: String::new(),
                generation: GenerationSettings::default(),
                sheet_width: sheet_width(),
                upscale_target: UpscaleTarget::default(),
            });
        }
        serde_json::from_slice(&std::fs::read(path)?).context("Could not read ui-settings.json.")
    }
    pub fn save(&self, project: &Path) -> Result<()> {
        let dir = project.join(".imagine");
        std::fs::create_dir_all(&dir)?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        let temp = dir.join(format!("ui-settings-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp)?;
            file.write_all(&serde_json::to_vec_pretty(self)?)?;
            file.sync_all()?;
            std::fs::rename(&temp, dir.join("ui-settings.json"))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temp);
        }
        result
    }
}
pub fn check_output_folder(folder: &Path) -> Result<()> {
    if !folder.is_absolute() {
        bail!("Choose an absolute path for the image folder.");
    }
    std::fs::create_dir_all(folder)
        .context("Could not create the image folder. Choose another folder.")?;
    let probe = folder.join(format!(".imagine-write-{}", uuid::Uuid::new_v4()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&probe)
        .context("Cannot write to the image folder. Choose another folder.")?;
    drop(file);
    std::fs::remove_file(probe)?;
    Ok(())
}
pub fn save_image(folder: &Path, bytes: &[u8]) -> Result<PathBuf> {
    check_output_folder(folder)?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    let target = folder.join(format!("imagine-{id}.png"));
    // create_new refuses overwrites, including on filesystems without hard links.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&target)
        .context("Could not create the image file. Choose another folder and retry.")?;
    let result = (|| -> Result<PathBuf> {
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(target.clone())
    })();
    drop(file);
    if result.is_err() {
        let _ = std::fs::remove_file(&target);
    }
    result.context(
        "The image is ready on the server but could not be saved. Choose a writable folder and retry.",
    )
}

/// Save one complete sheet set, cleaning up only files created by this attempt on failure.
pub fn save_sheet(folder: &Path, images: &[(&str, &[u8])]) -> Result<Vec<PathBuf>> {
    if images.iter().map(|(role, _)| *role).collect::<Vec<_>>()
        != ["character_sheet", "portrait", "turnaround"]
    {
        bail!("The character sheet result is incomplete.");
    }
    check_output_folder(folder)?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    let mut created = Vec::new();
    let result = (|| -> Result<()> {
        for (role, bytes) in images {
            let path = folder.join(format!("imagine-{id}-{role}.png"));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)?;
            created.push(path);
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        Ok(())
    })();
    if result.is_err() {
        let mut cleanup_failed = false;
        for path in &created {
            cleanup_failed |= std::fs::remove_file(path).is_err();
        }
        if cleanup_failed {
            bail!(
                "Could not save the complete sheet or remove partial files. Check the output folder before retrying."
            );
        }
    }
    result.context("The sheet is ready on the server but could not be saved. Choose a writable folder and retry.")?;
    Ok(created)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upscale_targets_follow_selected_ratio_with_aligned_short_edge() {
        for (target, long, wide, two_thirds) in [
            (UpscaleTarget::K2, 2048, 1152, 1376),
            (UpscaleTarget::K3, 3072, 1728, 2048),
            (UpscaleTarget::K4, 4096, 2304, 2720),
        ] {
            assert_eq!(target.dimensions("16:9").unwrap(), (long, wide));
            assert_eq!(target.dimensions("9:16").unwrap(), (wide, long));
            assert_eq!(target.dimensions("1:1").unwrap(), (long, long));
            assert_eq!(target.dimensions("2:3").unwrap(), (two_thirds, long));
            for ratio in ASPECT_RATIOS {
                let (w, h) = target.dimensions(ratio).unwrap();
                assert_eq!(w.max(h), long);
                assert!(w.is_multiple_of(32) && h.is_multiple_of(32));
            }
        }
        let root = tempfile::tempdir().unwrap();
        let old: Preferences =
            serde_json::from_value(serde_json::json!({"output_folder":root.path()})).unwrap();
        assert_eq!(old.upscale_target, UpscaleTarget::K2);
    }
    #[test]
    fn sheet_files_share_a_unique_prefix_and_keep_private_permissions() {
        let root = tempfile::tempdir().unwrap();
        let images = [
            ("character_sheet", b"sheet".as_slice()),
            ("portrait", b"portrait".as_slice()),
            ("turnaround", b"views".as_slice()),
        ];
        let files = save_sheet(root.path(), &images).unwrap();
        let again = save_sheet(root.path(), &images).unwrap();
        use std::os::unix::fs::PermissionsExt;
        for ((path, (_, bytes)), other) in files.iter().zip(images).zip(again) {
            assert_eq!(std::fs::read(path).unwrap(), bytes);
            assert_ne!(*path, other);
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(save_sheet(root.path(), &images[..2]).is_err());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 6);
        let old: Preferences =
            serde_json::from_value(serde_json::json!({"output_folder":root.path()})).unwrap();
        assert_eq!(old.sheet_width, 1536);
    }
    #[test]
    fn dimensions_match_aio_presets_and_ties_to_even() {
        // Values independently obtained from AIO services/dimensions.py.
        let expected = [
            (1600, 1600),
            (1056, 1600),
            (1600, 1056),
            (1216, 1600),
            (1600, 1216),
            (1280, 1600),
            (1600, 1280),
            (896, 1600),
            (1600, 896),
        ];
        for (ratio, dimensions) in ASPECT_RATIOS.iter().zip(expected) {
            let settings = GenerationSettings {
                aspect_ratio: (*ratio).into(),
                ..Default::default()
            };
            assert_eq!(settings.dimensions().unwrap(), dimensions);
        }
        for (max_side, height) in [(256, 128), (768, 448)] {
            assert_eq!(
                GenerationSettings {
                    max_side,
                    ..Default::default()
                }
                .dimensions()
                .unwrap(),
                (max_side, height)
            );
        }
        for max_side in [0, 255, 257, 4097, u32::MAX] {
            assert!(
                GenerationSettings {
                    max_side,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            GenerationSettings {
                aspect_ratio: "unknown".into(),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn settings_and_unique_files() {
        let root = tempfile::tempdir().unwrap();
        let p = Preferences {
            output_folder: root.path().join("pictures"),
            privacy_mode: false,
            negative_prompt: "  artifacts, suddig\nextra fingrar  ".into(),
            sheet_width: 1536,
            upscale_target: UpscaleTarget::K3,
            generation: GenerationSettings {
                steps: 24,
                cfg_scale: 3.,
                random_seed: false,
                seed: u32::MAX,
                aspect_ratio: "3:4".into(),
                max_side: 1024,
                reference_max_pixels_per_image: 2_500_000,
            },
        };
        p.save(root.path()).unwrap();
        assert_eq!(Preferences::load(root.path()).unwrap(), p);
        let a = save_image(&p.output_folder, b"first").unwrap();
        let b = save_image(&p.output_folder, b"second").unwrap();
        assert_ne!(a, b);
        assert_eq!(std::fs::read(a).unwrap(), b"first");
        assert!(check_output_folder(&b).is_err());
        assert!(check_output_folder(Path::new("relative")).is_err());
    }

    #[test]
    fn existing_settings_keep_folder_and_default_to_privacy_on() {
        let root = tempfile::tempdir().unwrap();
        let defaults = Preferences::load(root.path()).unwrap();
        assert!(defaults.privacy_mode);
        assert!(defaults.negative_prompt.is_empty());
        defaults.save(root.path()).unwrap();
        let folder = root.path().join("existing-pictures");
        std::fs::write(
            root.path().join(".imagine/ui-settings.json"),
            serde_json::to_vec(&serde_json::json!({"output_folder": folder})).unwrap(),
        )
        .unwrap();
        let loaded = Preferences::load(root.path()).unwrap();
        assert_eq!(loaded.output_folder, folder);
        assert!(loaded.privacy_mode);
        assert!(loaded.negative_prompt.is_empty());
        assert_eq!(loaded.generation, GenerationSettings::default());
        let previous: GenerationSettings = serde_json::from_value(serde_json::json!({
            "steps": 24, "max_side": 1024
        }))
        .unwrap();
        assert_eq!(previous.steps, 24);
        assert_eq!(previous.max_side, 1024);
        assert_eq!(previous.reference_max_pixels_per_image, 1_000_000);
    }
}
