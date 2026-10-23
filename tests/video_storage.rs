//! Synthetic fixtures only. Run with TMPDIR inside the approved task root.
#[path = "support/video_output.rs"]
mod support;
#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;
use imagine_server::{
    video::output::Destination,
    video_destination::{self as d, Config, Environment},
};
use std::sync::Arc;
fn own() -> tempfile::TempDir {
    let root = task_fixtures::env_root("TMPDIR");
    tempfile::tempdir_in(root).unwrap()
}
#[test]
fn persists_after_session_and_handles_parallel_names_without_overwriting() {
    let root = own();
    let dest = support::destination(root.path());
    let proof = serde_json::to_value(dest.proof().unwrap()).unwrap();
    let jobs = (0..6)
        .map(|i| {
            let d = dest.clone();
            let p = proof.clone();
            std::thread::spawn(move || {
                let mut pending = d.reserve(&p, 64).unwrap();
                let saved = pending.publish(&[i; 64]).unwrap();
                let path = saved.path.clone();
                pending.commit();
                (path, i)
            })
        })
        .collect::<Vec<_>>();
    let mut names = std::collections::HashSet::new();
    for job in jobs {
        let (path, i) = job.join().unwrap();
        assert!(names.insert(path.clone()));
        assert_eq!(std::fs::read(path).unwrap(), [i; 64]);
    }
    drop(dest);
    assert!(names.iter().all(|p| p.exists()));
}
#[test]
fn custom_video_names_preserve_unicode_do_not_overwrite_and_cannot_change_destination() {
    let root = own();
    let dest = support::destination(root.path());
    let proof = serde_json::to_value(dest.proof().unwrap()).unwrap();
    let mut a = dest.reserve(&proof, 64).unwrap();
    let first = a.publish_video(b"own first", Some("Min film β")).unwrap();
    a.commit();
    let mut b = dest.reserve(&proof, 64).unwrap();
    let second = b
        .publish_video(b"own second", Some("Min film β.mp4"))
        .unwrap();
    b.commit();
    assert_eq!(first.path.file_name().unwrap(), "Min film β.mp4");
    assert_eq!(second.path.file_name().unwrap(), "Min film β_2.mp4");
    assert_eq!(std::fs::read(&first.path).unwrap(), b"own first");
    assert_eq!(std::fs::read(&second.path).unwrap(), b"own second");
    let mut c = dest.reserve(&proof, 64).unwrap();
    let upper = c.publish_video(b"own upper", Some("Upper.MP4")).unwrap();
    c.commit();
    assert_eq!(upper.path.file_name().unwrap(), "Upper.MP4");
    for name in [
        "../outside.mp4",
        "/outside.mp4",
        "nested/outside.mp4",
        "nested/",
        "nested/.",
        "nul\0.mp4",
    ] {
        let mut p = dest.reserve(&proof, 64).unwrap();
        assert_eq!(
            p.publish_video(b"own invalid", Some(name)).err(),
            Some("invalid_output_filename")
        );
    }
}
#[test]
fn cancellation_and_failed_publication_remove_only_own_partial() {
    let root = own();
    let dest = support::destination(root.path());
    let proof = serde_json::to_value(dest.proof().unwrap()).unwrap();
    let mut first = dest.reserve(&proof, 64).unwrap();
    let saved = first.publish(b"previous own saved fixture").unwrap();
    let path = saved.path.clone();
    first.commit();
    let pending = dest.reserve(&proof, 64).unwrap();
    drop(pending);
    let mut failed = dest.reserve(&proof, 64).unwrap();
    let transient = failed.publish(b"own transaction pending").unwrap();
    let transient_path = transient.path.clone();
    drop(transient);
    drop(failed);
    assert!(!transient_path.exists());
    assert_eq!(std::fs::read(&path).unwrap(), b"previous own saved fixture");
    let listing = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(listing.len(), 1);
}
struct View {
    table: String,
}
impl Environment for View {
    fn mounts(&self) -> d::Result<String> {
        Ok(self.table.clone())
    }
    fn host(&self) -> d::Result<String> {
        Ok("own-synthetic-host".into())
    }
}
#[test]
fn missing_unmounted_readonly_wrong_volume_host_and_symlinks_fail_closed() {
    let root = own();
    let dest = support::destination(root.path());
    let identity = dest.proof().unwrap();
    let proof = serde_json::to_value(&identity).unwrap();
    let c = identity.config.clone();
    for (table, expected) in [
        (String::new(), "video_volume_not_mounted"),
        (
            format!("1 0 0:1 / {} ro - fixture fixture rw", c.mount.display()),
            "video_destination_read_only",
        ),
        (
            format!("1 0 0:1 / {} rw - fixture other rw", c.mount.display()),
            "video_volume_mismatch",
        ),
    ] {
        let d = Destination::with_environment(c.clone(), Arc::new(View { table }));
        assert_eq!(d.preflight(&proof, 1).unwrap_err(), expected);
    }
    let mut wrong = proof.clone();
    wrong["host"] = "other-machine".into();
    assert_eq!(
        dest.preflight(&wrong, 1).unwrap_err(),
        "video_destination_mismatch"
    );
    assert_eq!(
        dest.preflight(&proof, u64::MAX).unwrap_err(),
        "video_destination_full"
    );
    let missing = c.root.join("not-created");
    assert!(d::directory(&missing).is_err());
    assert!(!missing.exists());
    let link = root.path().join("link");
    std::os::unix::fs::symlink(&c.root, &link).unwrap();
    assert!(d::directory(&link).is_err());
}
#[test]
fn filesystem_identity_and_destination_permissions_are_checked() {
    use std::os::unix::fs::PermissionsExt;
    let root = own();
    let dest = support::destination(root.path());
    let identity = dest.proof().unwrap();
    let proof = serde_json::to_value(&identity).unwrap();
    let mut config: Config = identity.config.clone();
    config.fsid[0] ^= 1;
    let bad = Destination::with_environment(
        config,
        Arc::new(View {
            table: format!(
                "1 0 0:1 / {} rw - fixture fixture rw",
                root.path().display()
            ),
        }),
    );
    assert_eq!(bad.proof().unwrap_err(), "video_volume_mismatch");
    std::fs::set_permissions(
        &identity.config.root,
        std::fs::Permissions::from_mode(0o500),
    )
    .unwrap();
    let error = dest.preflight(&proof, 1).unwrap_err();
    assert_eq!(error, "video_destination_not_writable");
    std::fs::set_permissions(
        &identity.config.root,
        std::fs::Permissions::from_mode(0o700),
    )
    .unwrap();
}
#[tokio::test]
async fn unconfigured_destination_has_no_default_or_plaintext_fallback() {
    let root = own();
    let store = imagine_server::storage::Store::open(root.path())
        .await
        .unwrap();
    assert!(store.videos.destination().is_err());
    assert_eq!(std::fs::read_dir(&store.media).unwrap().count(), 0);
    store.close().await.unwrap();
}

#[test]
fn image_roles_share_number_and_cancel_keeps_previous_persistent_media() {
    let root = own();
    let dest = support::destination(root.path());
    let proof = serde_json::to_value(dest.proof().unwrap()).unwrap();
    let mut first = dest.reserve_image(&proof, 128, None).unwrap();
    let sheet = first
        .publish_image(b"own sheet", "character_sheet", "image/png", None)
        .unwrap();
    let group = sheet
        .path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .strip_suffix("_character_sheet")
        .unwrap();
    let mut second = dest.reserve_image(&proof, 128, None).unwrap();
    let portrait = second
        .publish_image(b"own portrait", "portrait", "image/png", Some(group))
        .unwrap();
    assert!(
        portrait
            .path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with(group)
    );
    first.commit();
    second.commit();
    let mut failed = dest.reserve_image(&proof, 128, None).unwrap();
    let partial = failed
        .publish_image(b"own failed", "image", "image/png", None)
        .unwrap();
    drop(failed);
    assert!(!partial.path.exists());
    assert_eq!(std::fs::read(&sheet.path).unwrap(), b"own sheet");
    assert_eq!(std::fs::read(&portrait.path).unwrap(), b"own portrait");
    let mut collision = dest.reserve_image(&proof, 128, None).unwrap();
    assert!(
        collision
            .publish_image(b"not overwrite", "portrait", "image/png", Some(group))
            .is_err()
    );
    assert_eq!(std::fs::read(&portrait.path).unwrap(), b"own portrait");
}
#[test]
fn simultaneous_image_names_are_unique_and_stay_on_destination() {
    let root = own();
    let dest = support::destination(root.path());
    let proof = serde_json::to_value(dest.proof().unwrap()).unwrap();
    let jobs = (0..8)
        .map(|n| {
            let d = dest.clone();
            let p = proof.clone();
            std::thread::spawn(move || {
                let mut pending = d.reserve_image(&p, 1024, None).unwrap();
                let saved = pending
                    .publish_image(&[n], "image", "image/png", None)
                    .unwrap();
                pending.commit();
                saved.path
            })
        })
        .collect::<Vec<_>>();
    let paths = jobs
        .into_iter()
        .map(|j| j.join().unwrap())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(paths.len(), 8);
    assert!(
        paths
            .iter()
            .all(|p| p.starts_with(root.path()) && p.exists())
    );
}
