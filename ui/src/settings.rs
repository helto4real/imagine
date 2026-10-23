use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs::OpenOptions,
    io::Write,
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
};

pub const DEFAULT_API_URL: &str = "http://127.0.0.1:8189/imagine";

#[derive(Clone)]
pub struct Connection {
    pub url: String,
}
impl Connection {
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
    #[serde(default = "default_model_group")]
    pub model_group: String,
    #[serde(default)]
    pub model_groups: std::collections::BTreeMap<String, ModelGroupSettings>,
    pub output_folder: PathBuf,
    #[serde(default = "privacy_default")]
    pub privacy_mode: bool,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default)]
    pub prompt_catalog_folder: Option<PathBuf>,
    #[serde(default)]
    pub preferred_prompt_profile: Option<String>,
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
    // None inherits the selected model's defaults, including older saved settings.
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
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
            sampler: None,
            scheduler: None,
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
        if self.steps == 0 || !self.cfg_scale.is_finite() {
            bail!("Steps must be positive and CFG finite.");
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

/// Explicit fixture storage is kept separate from normal HOME-based storage.
#[derive(Clone)]
pub enum PreferencesStore {
    Fixture(PathBuf),
    Home {
        paths: crate::storage_layout::PublicPaths,
        runtime: PathBuf,
        private: Option<std::sync::Arc<crate::private_storage::Root>>,
        private_error: Option<String>,
        system_private: bool,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct PublicUi {
    privacy_mode: bool,
    generation: GenerationSettings,
    sheet_width: u32,
    upscale_target: UpscaleTarget,
}
impl Default for PublicUi {
    fn default() -> Self {
        Self {
            privacy_mode: true,
            generation: Default::default(),
            sheet_width: sheet_width(),
            upscale_target: Default::default(),
        }
    }
}
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct PublicGroup {
    variant: String,
    loras: Vec<LoraSetting>,
}
#[derive(Serialize, Deserialize)]
struct Selection {
    model_group: String,
    model_groups: std::collections::BTreeMap<String, PublicGroup>,
}
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct PrivateUi {
    output_folder: PathBuf,
    negative_prompt: String,
    prompt_catalog_folder: Option<PathBuf>,
    preferred_prompt_profile: Option<String>,
    enhancement_profiles: std::collections::BTreeMap<String, String>,
}
const PRIVATE_UI: &str = "library/ui-preferences.json";
impl PreferencesStore {
    pub fn server_owned_output(&self) -> bool {
        matches!(self, Self::Home { .. })
    }
    pub fn system() -> Result<Self> {
        let home = PathBuf::from(std::env::var_os("HOME").context("HOME is unavailable")?);
        if !home.is_absolute() {
            bail!("HOME must be absolute")
        }
        let runtime = crate::storage_layout::runtime_directory(
            std::env::var_os("XDG_RUNTIME_DIR")
                .as_deref()
                .map(Path::new),
        )
        .map_err(anyhow::Error::msg)?;
        Ok(Self::Home {
            paths: crate::storage_layout::PublicPaths::for_home(&home),
            runtime,
            private: None,
            private_error: None,
            system_private: true,
        })
    }
    /// Explicit homes/volume providers make CPU/QA independent of actual HOME.
    pub fn for_home(
        home: &Path,
        runtime: PathBuf,
        private: Result<crate::private_storage::Root, &'static str>,
    ) -> Self {
        let (private, private_error) = match private {
            Ok(root) => (Some(std::sync::Arc::new(root)), None),
            Err(error) => (None, Some(error.into())),
        };
        Self::Home {
            paths: crate::storage_layout::PublicPaths::for_home(home),
            runtime,
            private,
            private_error,
            system_private: false,
        }
    }
    pub(crate) fn private_root(
        &self,
    ) -> std::result::Result<std::sync::Arc<crate::private_storage::Root>, String> {
        let Self::Home {
            private,
            private_error,
            system_private,
            ..
        } = self
        else {
            return Err("Explicit private store required".into());
        };
        if *system_private {
            // Reopening a normally mounted volume may recover a previous outage;
            // explicit CPU fixtures never inspect the system mount namespace.
            crate::private_storage::Root::system()
                .map(std::sync::Arc::new)
                .map_err(str::to_owned)
        } else {
            private.clone().ok_or_else(|| {
                private_error
                    .clone()
                    .unwrap_or_else(|| "private_storage_unavailable".into())
            })
        }
    }
    pub fn load(&self) -> Result<(Preferences, Option<String>)> {
        let Self::Home { paths, .. } = self else {
            let Self::Fixture(project) = self else {
                unreachable!()
            };
            return Preferences::load(project).map(|p| (p, None));
        };
        let public: PublicUi = crate::settings_store::read_value(&paths.ui)?
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();
        let selection: Option<Selection> =
            crate::settings_store::read_namespace(&paths.models, "selection")?
                .map(serde_json::from_value)
                .transpose()?;
        let (private_prefs, warning) = match self.private_root() {
            Ok(root) => match root.read(Path::new(PRIVATE_UI)) {
                Ok(bytes) => (
                    bytes
                        .map(|b| serde_json::from_slice::<PrivateUi>(&b))
                        .transpose()?,
                    None,
                ),
                Err(error) => (
                    None,
                    Some(format!("Private preferences unavailable: {error}")),
                ),
            },
            Err(error) => (
                None,
                Some(format!("Private preferences unavailable: {error}")),
            ),
        };
        let private_prefs = private_prefs.unwrap_or_default();
        let mut model_groups: std::collections::BTreeMap<String, ModelGroupSettings> = selection
            .as_ref()
            .map(|s| {
                s.model_groups
                    .iter()
                    .map(|(name, g)| {
                        (
                            name.clone(),
                            ModelGroupSettings {
                                variant: g.variant.clone(),
                                loras: g.loras.clone(),
                                enhancement_profile_id: private_prefs
                                    .enhancement_profiles
                                    .get(name)
                                    .cloned(),
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (name, profile) in &private_prefs.enhancement_profiles {
            model_groups
                .entry(name.clone())
                .or_default()
                .enhancement_profile_id = Some(profile.clone());
        }
        Ok((
            Preferences {
                model_group: selection.map_or_else(default_model_group, |s| s.model_group),
                model_groups,
                output_folder: private_prefs.output_folder,
                privacy_mode: public.privacy_mode,
                negative_prompt: private_prefs.negative_prompt,
                prompt_catalog_folder: private_prefs.prompt_catalog_folder,
                preferred_prompt_profile: private_prefs.preferred_prompt_profile,
                generation: public.generation,
                sheet_width: public.sheet_width,
                upscale_target: public.upscale_target,
            },
            warning,
        ))
    }
    pub fn save(&self, prefs: &Preferences) -> Result<()> {
        let Self::Home { paths, runtime, .. } = self else {
            let Self::Fixture(project) = self else {
                unreachable!()
            };
            return prefs.save(project);
        };
        let root = self
            .private_root()
            .map_err(|error| anyhow::anyhow!("Private preferences unavailable: {error}"))?;
        let private = PrivateUi {
            output_folder: prefs.output_folder.clone(),
            negative_prompt: prefs.negative_prompt.clone(),
            prompt_catalog_folder: prefs.prompt_catalog_folder.clone(),
            preferred_prompt_profile: prefs.preferred_prompt_profile.clone(),
            enhancement_profiles: prefs
                .model_groups
                .iter()
                .filter_map(|(name, g)| {
                    g.enhancement_profile_id
                        .as_ref()
                        .map(|id| (name.clone(), id.clone()))
                })
                .collect(),
        };
        // The private write must succeed before any public file reports this save.
        root.write(Path::new(PRIVATE_UI), &serde_json::to_vec_pretty(&private)?)
            .map_err(anyhow::Error::msg)?;
        let selection = Selection {
            model_group: prefs.model_group.clone(),
            model_groups: prefs
                .model_groups
                .iter()
                .map(|(name, g)| {
                    (
                        name.clone(),
                        PublicGroup {
                            variant: g.variant.clone(),
                            loras: g.loras.clone(),
                        },
                    )
                })
                .collect(),
        };
        crate::settings_store::write_namespace(
            &paths.models,
            "selection",
            serde_json::to_value(selection)?,
            runtime,
        )?;
        crate::settings_store::write_value(
            &paths.ui,
            serde_json::to_value(PublicUi {
                privacy_mode: prefs.privacy_mode,
                generation: prefs.generation.clone(),
                sheet_width: prefs.sheet_width,
                upscale_target: prefs.upscale_target,
            })?,
            runtime,
        )?;
        Ok(())
    }
}
impl Preferences {
    /// Explicit source-owned fixture API. Normal entry uses PreferencesStore::system.
    pub fn load(project: &Path) -> Result<Self> {
        let path = project.join(".imagine/ui-settings.json");
        if !path.exists() {
            return Ok(Self {
                model_group: default_model_group(),
                model_groups: Default::default(),
                output_folder: project.join(".imagine/images"),
                privacy_mode: true,
                negative_prompt: String::new(),
                prompt_catalog_folder: None,
                preferred_prompt_profile: None,
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
/// Uses the host's local calendar date, captured when the completed result is saved.
fn dated_folder(folder: &Path, now: std::time::SystemTime) -> Result<PathBuf> {
    let seconds = libc::time_t::try_from(now.duration_since(std::time::UNIX_EPOCH)?.as_secs())?;
    let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
    // localtime_r initializes caller-owned storage and uses the host timezone.
    let local = unsafe {
        if libc::localtime_r(&seconds, local.as_mut_ptr()).is_null() {
            bail!("Could not determine the local date for saving.");
        }
        local.assume_init()
    };
    Ok(folder.join(format!(
        "{:04}-{:02}-{:02}",
        local.tm_year + 1900,
        local.tm_mon + 1,
        local.tm_mday
    )))
}

pub fn save_image(folder: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let folder = dated_folder(folder, std::time::SystemTime::now())?;
    Ok(save_outputs(&folder, &[("", bytes)], Naming::Numbered)?.remove(0))
}

/// Save one complete sheet set with a shared number.
pub fn save_sheet(folder: &Path, images: &[(&str, &[u8])]) -> Result<Vec<PathBuf>> {
    if images.iter().map(|(role, _)| *role).collect::<Vec<_>>()
        != ["character_sheet", "portrait", "turnaround"]
    {
        bail!("The character sheet result is incomplete.");
    }
    let folder = dated_folder(folder, std::time::SystemTime::now())?;
    save_outputs(&folder, images, Naming::Numbered)
}

/// Keep variants beside their source, including across dates and output-folder changes.
pub fn save_upscaled(source: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let folder = source
        .parent()
        .context("The original image folder is unavailable.")?;
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .context("The original image filename is unavailable.")?;
    let base = match stem.rsplit_once("_upscaled") {
        Some((base, suffix))
            if suffix.is_empty()
                || suffix
                    .strip_prefix('_')
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit())) =>
        {
            base
        }
        _ => stem,
    };
    Ok(save_outputs(folder, &[("", bytes)], Naming::Upscaled(base))?.remove(0))
}

enum Naming<'a> {
    Numbered,
    Upscaled(&'a str),
}

fn save_outputs(
    folder: &Path,
    images: &[(&str, &[u8])],
    naming: Naming<'_>,
) -> Result<Vec<PathBuf>> {
    if !folder.is_absolute() {
        bail!("Choose an absolute path for the image folder.");
    }
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(folder)?;
    // Directory lock serializes number allocation across UI processes. Closing releases it;
    // no persistent lock file is needed. create_new also refuses external collisions/symlinks.
    let lock = std::fs::File::open(folder)?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error())
            .context("Could not reserve an image filename.");
    }
    let names = std::fs::read_dir(folder)?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let extension = "png";
    let prefix = "image_";
    let stem = match naming {
        Naming::Numbered => {
            let number = names
                .iter()
                .filter_map(|name| {
                    name.strip_prefix(prefix)?
                        .split(['_', '.'])
                        .next()?
                        .parse::<u64>()
                        .ok()
                })
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .context("Image numbering is exhausted.")?;
            format!("{prefix}{number:04}")
        }
        Naming::Upscaled(base) => {
            let prefix = format!("{base}_upscaled");
            let number = names
                .iter()
                .filter_map(|name| {
                    let suffix = name.strip_prefix(&prefix)?.strip_suffix(".png")?;
                    if suffix.is_empty() {
                        Some(1)
                    } else {
                        suffix.strip_prefix('_')?.parse::<u64>().ok()
                    }
                })
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .context("Upscale numbering is exhausted.")?;
            if number == 1 {
                prefix
            } else {
                format!("{prefix}_{number}")
            }
        }
    };
    let mut created = Vec::new();
    let result = (|| -> Result<()> {
        for (role, bytes) in images {
            let suffix = if role.is_empty() {
                String::new()
            } else {
                format!("_{role}")
            };
            let path = folder.join(format!("{stem}{suffix}.{extension}"));
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
                "Could not save the images or remove partial files. Check the output folder before retrying."
            );
        }
    }
    result.context("The result is ready on the server but could not be saved. Check the destination folder and retry.")?;
    Ok(created)
}

#[cfg(test)]
mod tests {
    #[test]
    fn image_sampling_has_no_quality_policy_ceiling() {
        let mut settings = super::GenerationSettings {
            steps: 130,
            cfg_scale: 20.1,
            ..Default::default()
        };
        settings.validate().unwrap();
        settings.cfg_scale = -1.;
        settings.validate().unwrap();
        settings.steps = 0;
        assert!(settings.validate().is_err());
        settings.steps = u32::MAX;
        settings.validate().unwrap();
        for cfg in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            settings.cfg_scale = cfg;
            assert!(settings.validate().is_err());
        }
    }

    use super::*;
    #[test]
    fn dated_numbering_groups_sheets_and_keeps_upscale_variants_beside_source() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::SystemTime::now();
        let today = dated_folder(root.path(), now).unwrap();
        let first = save_image(root.path(), b"original").unwrap();
        assert_eq!(first, today.join("image_0001.png"));
        let date = today.file_name().unwrap().to_str().unwrap();
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        assert_eq!(&date[7..8], "-");
        let images = [
            ("character_sheet", b"sheet".as_slice()),
            ("portrait", b"portrait".as_slice()),
            ("turnaround", b"views".as_slice()),
        ];
        let sheet = save_sheet(root.path(), &images).unwrap();
        for (file, (role, _)) in sheet.iter().zip(images) {
            assert_eq!(*file, today.join(format!("image_0002_{role}.png")));
        }
        for number in 3..=12 {
            let file = save_image(root.path(), b"next").unwrap();
            assert_eq!(file, today.join(format!("image_{number:04}.png")));
        }
        let mut names: Vec<_> = std::fs::read_dir(&today)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names[0], "image_0001.png");
        assert_eq!(names.last().unwrap(), "image_0012.png");
        // A previous date keeps its own sequence and all upscale variants.
        let yesterday = root.path().join("2000-01-01");
        let old = save_outputs(&yesterday, &[("", b"old")], Naming::Numbered)
            .unwrap()
            .remove(0);
        assert_eq!(old.file_name().unwrap(), "image_0001.png");
        let upscaled = save_upscaled(&old, b"upscaled").unwrap();
        assert_eq!(upscaled, yesterday.join("image_0001_upscaled.png"));
        let again = save_upscaled(&upscaled, b"variant").unwrap();
        assert_eq!(again, yesterday.join("image_0001_upscaled_2.png"));
        let third = save_upscaled(&old, b"another").unwrap();
        assert_eq!(third, yesterday.join("image_0001_upscaled_3.png"));
        assert_eq!(std::fs::read(&old).unwrap(), b"old");
        assert_eq!(std::fs::read(&upscaled).unwrap(), b"upscaled");
        assert_eq!(
            save_upscaled(&sheet[0], b"big sheet").unwrap(),
            today.join("image_0002_character_sheet_upscaled.png")
        );
        // Existing outputs, including partial sets, reserve their numbers after restart.
        std::fs::write(today.join("image_0020_portrait.png"), b"partial").unwrap();
        assert_eq!(
            save_image(root.path(), b"new session").unwrap(),
            today.join("image_0021.png")
        );
    }

    #[test]
    fn simultaneous_saves_allocate_distinct_numbers_without_overwriting() {
        let root = tempfile::tempdir().unwrap();
        let files = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..12u8)
                .map(|n| {
                    let folder = root.path();
                    scope.spawn(move || (n, save_image(folder, &[n]).unwrap()))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut names = std::collections::BTreeSet::new();
        for (n, file) in &files {
            assert_eq!(std::fs::read(file).unwrap(), [*n]);
            assert_eq!(
                std::fs::metadata(file).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert!(names.insert(file.file_name().unwrap().to_owned()));
        }
        assert_eq!(names.first().unwrap(), "image_0001.png");
        assert_eq!(names.last().unwrap(), "image_0012.png");
        let variants = std::thread::scope(|scope| {
            let source = &files[0].1;
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(move || save_upscaled(source, b"variant").unwrap()))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<std::collections::BTreeSet<_>>()
        });
        assert_eq!(variants.len(), 4);
    }

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
        assert_eq!(
            std::fs::read_dir(files[0].parent().unwrap())
                .unwrap()
                .count(),
            6
        );
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
            model_group: default_model_group(),
            model_groups: Default::default(),
            output_folder: root.path().join("pictures"),
            privacy_mode: false,
            negative_prompt: "  artifacts, suddig\nextra fingrar  ".into(),
            prompt_catalog_folder: Some(root.path().join("own-catalog")),
            preferred_prompt_profile: Some("own custom profile".into()),
            sheet_width: 1536,
            upscale_target: UpscaleTarget::K3,
            generation: GenerationSettings {
                steps: 24,
                cfg_scale: 3.,
                sampler: Some("euler".into()),
                scheduler: Some("simple".into()),
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

fn default_model_group() -> String {
    crate::model_catalog::QWEN_GROUP.into()
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelGroupSettings {
    pub variant: String,
    pub loras: Vec<LoraSetting>,
    pub enhancement_profile_id: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraSetting {
    pub id: String,
    pub name: String,
    pub strength: f32,
    pub enabled: bool,
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::{private_storage::Root, video_destination as d};
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    struct Volume(Mutex<String>);
    impl d::Environment for Volume {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            Ok("own-ui-settings-fixture".into())
        }
    }
    fn own() -> tempfile::TempDir {
        let root = crate::test_destination::task_fixtures::env_root("TMPDIR");
        tempfile::Builder::new()
            .prefix("cp42settings-")
            .tempdir_in(root)
            .unwrap()
    }
    fn volume(path: &Path) -> (Root, Arc<Volume>) {
        let mount = path.join("volume");
        let root = mount.join("imagine");
        std::fs::create_dir_all(&root).unwrap();
        let env = Arc::new(Volume(Mutex::new(format!(
            "1 0 0:1 / {} rw - fixturefs own-ui-settings rw\n",
            mount.display()
        ))));
        let config = d::Config {
            root,
            mount: mount.clone(),
            source: "own-ui-settings".into(),
            filesystem: "fixturefs".into(),
            fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
        };
        (Root::with_environment(config, env.clone()).unwrap(), env)
    }
    #[test]
    fn split_preferences_never_leak_private_fields_or_read_legacy_and_preserve_catalog() {
        let own = own();
        let home = own.path().join("home");
        std::fs::create_dir_all(home.join(".imagine")).unwrap();
        let legacy = home.join(".imagine/ui-settings.json");
        std::fs::write(
            &legacy,
            b"Own deliberately invalid legacy fixture; must not be read",
        )
        .unwrap();
        let (private, env) = volume(own.path());
        let runtime = own.path().join("runtime/imagine");
        let store = PreferencesStore::for_home(&home, runtime.clone(), Ok(private.clone()));
        let (initial, warning) = store.load().unwrap();
        assert!(warning.is_none());
        assert!(initial.output_folder.as_os_str().is_empty());
        let paths = crate::storage_layout::PublicPaths::for_home(&home);
        crate::settings_store::write_namespace(
            &paths.models,
            "catalog",
            json!({"root":"own-models"}),
            &runtime,
        )
        .unwrap();
        let mut prefs = initial;
        prefs.negative_prompt = "OWN PRIVATE NEGATIVE".into();
        prefs.output_folder = private.layout().root().join("projects/own/outputs/images");
        prefs.prompt_catalog_folder = Some(private.layout().library().join("OWN PRIVATE CATALOG"));
        prefs.preferred_prompt_profile = Some("OWN PRIVATE PROMPT PROFILE".into());
        prefs.model_groups.insert(
            prefs.model_group.clone(),
            ModelGroupSettings {
                variant: "own-model-id".into(),
                loras: vec![LoraSetting {
                    id: "own-lora-id".into(),
                    name: "own-lora".into(),
                    strength: 0.8,
                    enabled: true,
                }],
                enhancement_profile_id: Some("OWN PRIVATE PROVIDER PROFILE".into()),
            },
        );
        store.save(&prefs).unwrap();
        assert_eq!(store.load().unwrap(), (prefs.clone(), None));
        assert_eq!(
            crate::settings_store::read_namespace(&paths.models, "catalog").unwrap(),
            Some(json!({"root":"own-models"}))
        );
        for path in [&paths.ui, &paths.models] {
            let bytes = std::fs::read(path).unwrap();
            let text = String::from_utf8(bytes).unwrap();
            for private in [
                "OWN PRIVATE",
                "negative_prompt",
                "output_folder",
                "prompt_catalog_folder",
                "preferred_prompt_profile",
                "enhancement_profile_id",
            ] {
                assert!(!text.contains(private), "private field leaked: {private}");
            }
        }
        let private_bytes = private.read(Path::new(PRIVATE_UI)).unwrap().unwrap();
        assert!(
            String::from_utf8(private_bytes)
                .unwrap()
                .contains("OWN PRIVATE NEGATIVE")
        );
        assert_eq!(
            std::fs::read(&legacy).unwrap(),
            b"Own deliberately invalid legacy fixture; must not be read"
        );
        let public_before = std::fs::read(&paths.ui).unwrap();
        let models_before = std::fs::read(&paths.models).unwrap();
        env.0.lock().unwrap().clear();
        prefs.negative_prompt = "OWN UNSAVED RAM VALUE".into();
        assert!(store.save(&prefs).is_err());
        assert_eq!(prefs.negative_prompt, "OWN UNSAVED RAM VALUE");
        assert_eq!(std::fs::read(&paths.ui).unwrap(), public_before);
        assert_eq!(std::fs::read(&paths.models).unwrap(), models_before);
        let (_, warning) = store.load().unwrap();
        assert!(warning.unwrap().contains("Private preferences unavailable"));
    }
    #[test]
    fn unavailable_private_root_keeps_public_loadable_but_never_claims_save_success() {
        let own = own();
        let home = own.path().join("home");
        let runtime = own.path().join("runtime/imagine");
        let store = PreferencesStore::for_home(&home, runtime, Err("video_volume_not_mounted"));
        let (mut prefs, warning) = store.load().unwrap();
        assert!(warning.unwrap().contains("video_volume_not_mounted"));
        assert!(prefs.output_folder.as_os_str().is_empty());
        prefs.negative_prompt = "OWN RAM ONLY".into();
        assert!(store.save(&prefs).is_err());
        assert_eq!(prefs.negative_prompt, "OWN RAM ONLY");
        assert!(!home.join(".config").exists());
    }
}
