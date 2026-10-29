//! Own encrypted fixture adapter; production normalization is shared with authoring.
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let own =
        std::path::Path::new("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e");
    assert!(
        args.len() == 2
            && std::path::Path::new(&args[0]).starts_with(own)
            && std::path::Path::new(&args[1]).starts_with(own)
    );
    let input = std::fs::read(&args[0]).unwrap();
    let out = std::path::Path::new(&args[1]);
    match imagine_server::video::authoring::vision::normalize(&input) {
        Ok(result) => {
            let jpeg = imagine_server::video::authoring::vision::director_jpeg(&result).unwrap();
            std::fs::write(out.join("rust-rgb.raw"), result.rgb).unwrap();
            std::fs::write(out.join("rust.jpg"), jpeg).unwrap();
            println!(
                "{}",
                serde_json::json!({"width":result.width,"height":result.height})
            );
        }
        Err(code) => {
            println!("{}", serde_json::json!({"error":code}));
            std::process::exit(1);
        }
    }
}
