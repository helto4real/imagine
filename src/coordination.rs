//! Process-local job tickets carried over a private Unix socket. Loss is fatal
//! to the active operation; coordinated execution never falls back to standalone.
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::Path;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{
        UnixStream,
        unix::{OwnedReadHalf, OwnedWriteHalf},
    },
};
pub struct Ticket {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
}
impl Ticket {
    pub async fn register(socket: &Path, job: &str, bytes: usize) -> Result<Self> {
        let stream = UnixStream::connect(socket)
            .await
            .context("coordinator unavailable")?;
        let (reader, writer) = stream.into_split();
        let mut this = Self {
            reader: BufReader::new(reader),
            writer,
        };
        this.send(json!({"job":job,"bytes":bytes})).await?;
        let reply = this.receive().await?;
        if reply["queued"] != true {
            bail!("coordinator rejected job")
        };
        Ok(this)
    }
    pub async fn register_many(socket: &Path, jobs: &[(String, usize)]) -> Result<Vec<Self>> {
        if jobs.len() == 1 {
            return Ok(vec![Self::register(socket, &jobs[0].0, jobs[0].1).await?]);
        }
        let stream = UnixStream::connect(socket).await?;
        let (reader, writer) = stream.into_split();
        let mut allocation = Self {
            reader: BufReader::new(reader),
            writer,
        };
        allocation.send(json!({"jobs":jobs.iter().map(|(id,bytes)|json!({"job":id,"bytes":bytes})).collect::<Vec<_>>()})).await?;
        let reply = allocation.receive().await?;
        if reply["queued"] != true {
            bail!("coordinator rejected batch")
        }
        let tokens = reply["tickets"]
            .as_array()
            .filter(|v| v.len() == jobs.len())
            .context("invalid batch tickets")?;
        let mut tickets = Vec::new();
        for token in tokens {
            let stream = UnixStream::connect(socket).await?;
            let (reader, writer) = stream.into_split();
            let mut ticket = Self {
                reader: BufReader::new(reader),
                writer,
            };
            ticket.send(json!({"ticket":token})).await?;
            if ticket.receive().await?["queued"] != true {
                bail!("coordinator rejected ticket")
            };
            tickets.push(ticket);
        }
        Ok(tickets)
    }
    async fn send(&mut self, v: Value) -> Result<()> {
        self.writer.write_all(format!("{v}\n").as_bytes()).await?;
        Ok(())
    }
    async fn receive(&mut self) -> Result<Value> {
        let mut line = String::new();
        if self.reader.read_line(&mut line).await? == 0 {
            bail!("coordinator disconnected")
        };
        Ok(serde_json::from_str(&line)?)
    }
    pub async fn acquire(mut self) -> Result<Self> {
        let reply = self.receive().await?;
        if reply["granted"] != true {
            bail!("coordinator did not grant GPU")
        };
        Ok(self)
    }
    pub async fn lost(&mut self) {
        let _ = self.reader.read_u8().await;
    }
    pub async fn finish(mut self) -> Result<()> {
        self.send(json!({"finished":true})).await
    }
}
