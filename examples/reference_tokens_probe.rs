//! Pure source-token oracle adapter; no models, providers or filesystem media.
fn main() {
    let values: Vec<String> = serde_json::from_reader(std::io::stdin()).unwrap();
    let aliases = values
        .iter()
        .map(|s| imagine_video_contract::references::aliases(s))
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string(&aliases).unwrap());
}
