//! Session-local presentation of jobs admitted to the server's shared FIFO.
use super::*;

#[derive(Clone)]
pub(super) struct Snapshot {
    pub prompt: String,
    pub preferences: Preferences,
    pub references: Vec<ReferenceView>,
}

pub(super) struct Job {
    pub id: String,
    pub batch: String,
    pub submission: Submission,
    pub snapshot: Snapshot,
    pub status: String,
    pub working: bool,
    pub cancelling: bool,
    pub retryable: bool,
    pub error: Option<String>,
    pub sampling: Option<client::SamplingProgress>,
    started: std::time::Instant,
}

pub(super) struct GalleryItem {
    pub id: String,
    /// These textures are small thumbnails; only the main viewer owns full pixels.
    pub result: Results,
    pub snapshot: Option<Snapshot>,
    pub elapsed: Option<Duration>,
}

pub(super) struct Queue {
    pub jobs: Vec<Job>,
    pub gallery: Vec<GalleryItem>,
    pub selected: Option<String>,
    pub loading: Option<String>,
    pub follow: bool,
    pub collapsed: bool,
    pub drafts: Option<Vec<String>>,
    pub draft_error: Option<String>,
}
impl Default for Queue {
    fn default() -> Self {
        Self {
            jobs: Vec::new(),
            gallery: Vec::new(),
            selected: None,
            loading: None,
            follow: true,
            collapsed: false,
            drafts: None,
            draft_error: None,
        }
    }
}
impl Queue {
    pub fn stage(&mut self, prepared: Vec<(Submission, Snapshot)>) -> String {
        let batch = uuid::Uuid::new_v4().to_string();
        for (submission, snapshot) in prepared {
            self.jobs.push(Job {
                id: submission.key.clone(),
                batch: batch.clone(),
                submission,
                snapshot,
                status: "Submitting".into(),
                working: true,
                cancelling: false,
                retryable: true,
                error: None,
                sampling: None,
                started: std::time::Instant::now(),
            });
        }
        batch
    }
}

pub(super) struct Completion {
    pub done: Completed,
    pub thumbnails: Vec<Preview>,
}
impl Completion {
    pub fn new(done: Completed) -> Self {
        let thumbnails = done
            .assets
            .iter()
            .map(|asset| client::thumbnail(&asset.preview))
            .collect();
        Self { done, thumbnails }
    }
}
pub(super) enum Event {
    Accepted(Vec<(String, String)>),
    AdmissionFailed(Vec<String>, Failure),
    Progress(String, Progress),
    Finished(String, Result<Completion, Failure>),
    Cancelled(String, String, Result<(), Failure>),
    Loaded(String, Result<Completed, String>),
}

impl ImagineApp {
    pub(super) fn archive_completed(&mut self, ctx: &egui::Context, done: &Completed) {
        let id = uuid::Uuid::new_v4().to_string();
        let result = Results {
            seed: done.seed,
            assets: done
                .assets
                .iter()
                .map(|asset| ResultView {
                    asset_id: asset.asset_id.clone(),
                    role: asset.role,
                    file: asset.file.clone(),
                    texture: texture(
                        ctx,
                        &client::thumbnail(&asset.preview),
                        &format!("thumb-{id}"),
                    ),
                })
                .collect(),
        };
        self.queue.gallery.push(GalleryItem {
            id: id.clone(),
            result,
            snapshot: None,
            elapsed: None,
        });
        self.queue.selected = Some(id);
        self.queue.loading = None;
    }

    pub(super) fn prepare_prompts(
        &self,
        prompts: &[String],
    ) -> anyhow::Result<Vec<(Submission, Snapshot)>> {
        if prompts.is_empty() || prompts.iter().any(|prompt| prompt.trim().is_empty()) {
            anyhow::bail!("Enter a prompt in every row.");
        }
        let metadata = self
            .metadata
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Reconnect to load model settings."))?;
        let references: Vec<_> = self.image.references.iter().map(|r| r.id.clone()).collect();
        // Validate the entire series before creating any queue rows or admitting work.
        prompts
            .iter()
            .map(|prompt| {
                let request = self.apply_model_selection(client::make_request(
                    prompt,
                    &self.prefs.negative_prompt,
                    &references,
                    metadata,
                    &self.prefs.generation,
                )?)?;
                Ok((
                    Submission {
                        upscale_source: None,
                        request,
                        key: uuid::Uuid::new_v4().to_string(),
                        folder: self.prefs.output_folder.clone(),
                        job: None,
                    },
                    Snapshot {
                        prompt: prompt.clone(),
                        preferences: self.prefs.clone(),
                        references: self.image.references.clone(),
                    },
                ))
            })
            .collect()
    }

    pub(super) fn enqueue_prompts(&mut self, ctx: &egui::Context, prompts: Vec<String>) -> bool {
        if self.client.is_none() || !self.editable() || self.picking_folder {
            return false;
        }
        let prepared = match self.prepare_prompts(&prompts) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.image.error = Some(error.to_string());
                self.queue.draft_error = Some(error.to_string());
                return false;
            }
        };
        let batch = self.queue.stage(prepared);
        self.image.error = None;
        self.queue.draft_error = None;
        self.queue.collapsed = false;
        self.run_batch(ctx, &batch);
        true
    }

    fn run_batch(&self, ctx: &egui::Context, batch: &str) {
        let Some(client) = self.client.clone() else {
            return;
        };
        let jobs: Vec<_> = self
            .queue
            .jobs
            .iter()
            .filter(|j| j.batch == batch)
            .map(|j| (j.id.clone(), j.submission.clone()))
            .collect();
        let batch = batch.to_owned();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            runtime().block_on(async {
                let submissions: Vec<_> = jobs.iter().map(|(_, s)| s.clone()).collect();
                let ids = match client.submit_batch(&submissions, &batch).await {
                    Ok(ids) => ids,
                    Err(error) => {
                        send(
                            &tx,
                            &ctx,
                            Message::Queue(Event::AdmissionFailed(
                                jobs.into_iter().map(|(id, _)| id).collect(),
                                error,
                            )),
                        );
                        return;
                    }
                };
                send(
                    &tx,
                    &ctx,
                    Message::Queue(Event::Accepted(
                        jobs.iter()
                            .zip(&ids)
                            .map(|((id, _), job)| (id.clone(), job.clone()))
                            .collect(),
                    )),
                );
                let mut tasks = tokio::task::JoinSet::new();
                for ((id, mut submission), job) in jobs.into_iter().zip(ids) {
                    submission.job = Some(job);
                    let (client, tx, ctx) = (client.clone(), tx.clone(), ctx.clone());
                    tasks.spawn(async move {
                        let result = client
                            .generate(submission, |p| {
                                send(&tx, &ctx, Message::Queue(Event::Progress(id.clone(), p)))
                            })
                            .await;
                        send(
                            &tx,
                            &ctx,
                            Message::Queue(Event::Finished(id, result.map(Completion::new))),
                        );
                    });
                }
                while tasks.join_next().await.is_some() {}
            });
        });
    }

    pub(super) fn retry_queue_job(&mut self, ctx: &egui::Context, id: &str) {
        let Some(index) = self
            .queue
            .jobs
            .iter()
            .position(|job| job.id == id && !job.working && job.error.is_some())
        else {
            return;
        };
        if self.client.is_none() || self.connecting {
            return;
        }
        if self.queue.jobs[index].submission.job.is_none() {
            let batch = self.queue.jobs[index].batch.clone();
            // Uncertain admission must replay the exact same batch and key.
            if self.queue.jobs[index].retryable {
                for job in self.queue.jobs.iter_mut().filter(|job| job.batch == batch) {
                    job.working = true;
                    job.error = None;
                    job.status = "Submitting".into();
                    job.submission.folder = self.prefs.output_folder.clone();
                }
                self.run_batch(ctx, &batch);
                return;
            }
        }
        let job = &mut self.queue.jobs[index];
        job.submission.folder = self.prefs.output_folder.clone();
        if !job.retryable {
            // A confirmed terminal failure is an explicit new generation.
            job.submission.job = None;
            job.submission.key = uuid::Uuid::new_v4().to_string();
        }
        job.error = None;
        job.cancelling = false;
        job.working = true;
        job.sampling = None;
        job.started = std::time::Instant::now();
        if job.submission.job.is_none() {
            job.batch = uuid::Uuid::new_v4().to_string();
            job.status = "Submitting".into();
            let batch = job.batch.clone();
            self.run_batch(ctx, &batch);
        } else {
            job.status = "Retrieving".into();
            let (id, submission) = (job.id.clone(), job.submission.clone());
            let (client, tx, ctx) = (self.client.clone().unwrap(), self.tx.clone(), ctx.clone());
            std::thread::spawn(move || {
                let result = runtime().block_on(client.generate(submission, |p| {
                    send(&tx, &ctx, Message::Queue(Event::Progress(id.clone(), p)))
                }));
                send(
                    &tx,
                    &ctx,
                    Message::Queue(Event::Finished(id, result.map(Completion::new))),
                );
            });
        }
    }

    pub(super) fn cancel_queue_job(&mut self, ctx: &egui::Context, id: &str) {
        let Some(job) = self
            .queue
            .jobs
            .iter_mut()
            .find(|j| j.id == id && j.working && !j.cancelling)
        else {
            return;
        };
        let (Some(client), Some(server_id)) = (self.client.clone(), job.submission.job.clone())
        else {
            return;
        };
        job.cancelling = true;
        job.status = "Cancelling".into();
        let (id, tx, ctx) = (id.to_owned(), self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let result = runtime().block_on(client.cancel(&server_id));
            send(
                &tx,
                &ctx,
                Message::Queue(Event::Cancelled(id, server_id, result)),
            );
        });
    }

    pub(super) fn queue_message(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::Accepted(ids) => {
                for (id, server_id) in ids {
                    if let Some(job) = self.queue.jobs.iter_mut().find(|j| j.id == id) {
                        job.submission.job = Some(server_id);
                        job.status = "Queued".into();
                    }
                }
            }
            Event::AdmissionFailed(ids, error) => {
                for job in &mut self.queue.jobs {
                    if ids.contains(&job.id) {
                        job.working = false;
                        job.status = "Failed".into();
                        job.retryable = error.retryable;
                        job.error = Some(error.message.clone());
                    }
                }
            }
            Event::Progress(id, progress) => {
                if let Some(job) = self.queue.jobs.iter_mut().find(|j| j.id == id && j.working) {
                    match progress {
                        Progress::Submitted(server_id) => job.submission.job = Some(server_id),
                        Progress::Status { label, sampling } => {
                            if !job.cancelling {
                                job.status = label;
                            }
                            job.sampling = sampling;
                        }
                    }
                }
            }
            Event::Finished(id, result) => {
                let Some(job) = self.queue.jobs.iter_mut().find(|j| j.id == id && j.working) else {
                    return;
                };
                job.working = false;
                job.sampling = None;
                job.cancelling = false;
                match result {
                    Err(error) => {
                        let cancelled = error.message == "Generation cancelled.";
                        job.status = if cancelled { "Cancelled" } else { "Failed" }.into();
                        job.retryable = error.retryable;
                        job.error = if cancelled { None } else { Some(error.message) };
                    }
                    Ok(completed) => {
                        job.status = "Saved".into();
                        job.error = None;
                        let elapsed = Some(job.started.elapsed());
                        let snapshot = Some(job.snapshot.clone());
                        let thumbs = completed
                            .done
                            .assets
                            .iter()
                            .zip(&completed.thumbnails)
                            .map(|(asset, preview)| ResultView {
                                asset_id: asset.asset_id.clone(),
                                role: asset.role,
                                file: asset.file.clone(),
                                texture: texture(ctx, preview, &format!("thumb-{id}")),
                            })
                            .collect();
                        self.queue.gallery.push(GalleryItem {
                            id: id.clone(),
                            result: Results {
                                assets: thumbs,
                                seed: completed.done.seed,
                            },
                            snapshot,
                            elapsed,
                        });
                        if self.queue.follow
                            && self.workspace == Workspace::Image
                            && self.editable()
                            && !self.edit.open
                        {
                            self.show_queue_result(ctx, id, completed.done, elapsed);
                        }
                    }
                }
            }
            Event::Cancelled(id, server_id, result) => {
                if let Some(job) = self.queue.jobs.iter_mut().find(|j| {
                    j.id == id && j.working && j.submission.job.as_ref() == Some(&server_id)
                }) && let Err(error) = result
                {
                    job.cancelling = false;
                    job.error = Some(error.message);
                }
            }
            Event::Loaded(id, result) => {
                if self.queue.loading.as_deref() != Some(&id) {
                    return;
                }
                self.queue.loading = None;
                match result {
                    Ok(done) if !self.busy() => {
                        let elapsed = self
                            .queue
                            .gallery
                            .iter()
                            .find(|r| r.id == id)
                            .and_then(|r| r.elapsed);
                        self.show_queue_result(ctx, id, done, elapsed);
                    }
                    Err(error) => self.image.error = Some(error),
                    _ => {}
                }
            }
        }
    }

    fn show_queue_result(
        &mut self,
        ctx: &egui::Context,
        id: String,
        done: Completed,
        elapsed: Option<Duration>,
    ) {
        self.queue.loading = None;
        self.queue.selected = Some(id);
        self.image.result = Some(Results {
            assets: done
                .assets
                .into_iter()
                .map(|asset| ResultView {
                    texture: texture(ctx, &asset.preview, &asset.asset_id),
                    asset_id: asset.asset_id,
                    role: asset.role,
                    file: asset.file,
                })
                .collect(),
            seed: done.seed,
        });
        self.image.selected_role = OutputRole::Image;
        self.image.actual_pixels = false;
        self.image.image_pan = egui::Vec2::ZERO;
        self.image.last_elapsed = elapsed;
        self.image.status = "Saved".into();
        self.image.error = None;
        self.edit = editing::EditSession::default();
        self.workspace = Workspace::Image;
    }

    pub(super) fn select_queue_result(&mut self, ctx: &egui::Context, id: &str) {
        if self.busy() || self.image.pending.is_some() {
            return;
        }
        let Some(item) = self.queue.gallery.iter().find(|r| r.id == id) else {
            return;
        };
        let result = item.result.clone();
        self.queue.follow = false;
        self.queue.loading = Some(id.to_owned());
        let (id, tx, ctx) = (id.to_owned(), self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            let done = (|| -> anyhow::Result<Completed> {
                Ok(Completed {
                    seed: result.seed,
                    assets: result
                        .assets
                        .into_iter()
                        .map(|asset| {
                            Ok(client::CompletedAsset {
                                preview: client::load_preview(&asset.file)?,
                                asset_id: asset.asset_id,
                                role: asset.role,
                                file: asset.file,
                            })
                        })
                        .collect::<anyhow::Result<_>>()?,
                })
            })()
            .map_err(|_| {
                "Could not open the saved image. Check that the output file is still available."
                    .into()
            });
            send(&tx, &ctx, Message::Queue(Event::Loaded(id, done)));
        });
    }

    pub(super) fn reuse_queue_settings(&mut self, id: &str) {
        if !self.editable() {
            return;
        }
        let Some(snapshot) = self
            .queue
            .gallery
            .iter()
            .find(|r| r.id == id)
            .and_then(|r| r.snapshot.clone())
        else {
            return;
        };
        self.image.prompt = snapshot.prompt;
        self.image.references = snapshot.references;
        self.prefs.negative_prompt = snapshot.preferences.negative_prompt;
        self.prefs.generation = snapshot.preferences.generation;
        self.prefs.model_group = snapshot.preferences.model_group;
        let group_settings = snapshot
            .preferences
            .model_groups
            .get(&self.prefs.model_group)
            .cloned()
            .unwrap_or_default();
        self.prefs
            .model_groups
            .insert(self.prefs.model_group.clone(), group_settings);
        if self.models.catalog.is_some() {
            self.metadata = self.selected_model().cloned();
        }
        self.workspace = Workspace::Image;
        self.save_preferences();
    }
}
