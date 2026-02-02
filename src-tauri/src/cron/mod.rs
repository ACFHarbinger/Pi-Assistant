use anyhow::Result;
use chrono_tz::Tz;
use pi_core::agent_types::AgentCommand;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: Uuid,
    pub schedule: String,
    pub task_description: String,
    #[serde(default)]
    pub timezone: Option<String>,
}

pub struct CronManager {
    scheduler: JobScheduler,
    jobs: Arc<Mutex<HashMap<Uuid, CronJob>>>,
    scheduler_ids: Arc<Mutex<HashMap<Uuid, Uuid>>>, // Maps our ID to scheduler's ID
    config_path: PathBuf,
    agent_cmd_tx: mpsc::Sender<AgentCommand>,
}

impl CronManager {
    pub async fn new(config_dir: &Path, agent_cmd_tx: mpsc::Sender<AgentCommand>) -> Result<Self> {
        let scheduler = JobScheduler::new().await?;
        let config_path = config_dir.join("cron.json");
        let jobs = Arc::new(Mutex::new(HashMap::new()));
        let scheduler_ids = Arc::new(Mutex::new(HashMap::new()));

        let manager = Self {
            scheduler,
            jobs,
            scheduler_ids,
            config_path,
            agent_cmd_tx,
        };

        manager.load_jobs().await?;
        manager.scheduler.start().await?;

        Ok(manager)
    }

    async fn load_jobs(&self) -> Result<()> {
        if !self.config_path.exists() {
            return Ok(());
        }

        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let jobs: Vec<CronJob> = serde_json::from_str(&content)?;

        for job in jobs {
            if let Err(e) = self.add_job_to_scheduler_internal(job).await {
                error!("Failed to load cron job: {}", e);
            }
        }

        Ok(())
    }

    async fn save_jobs(&self) -> Result<()> {
        let jobs: Vec<CronJob> = self.jobs.lock().await.values().cloned().collect();
        let content = serde_json::to_string_pretty(&jobs)?;
        tokio::fs::write(&self.config_path, content).await?;
        Ok(())
    }

    pub async fn add_job(
        &self,
        schedule: String,
        task_description: String,
        timezone: Option<String>,
    ) -> Result<Uuid> {
        let job = CronJob {
            id: Uuid::new_v4(),
            schedule,
            task_description,
            timezone,
        };

        self.add_job_to_scheduler_internal(job.clone()).await?;
        self.save_jobs().await?;

        Ok(job.id)
    }

    async fn add_job_to_scheduler_internal(&self, job: CronJob) -> Result<()> {
        let our_id = job.id;
        let schedule_str = job.schedule.clone();
        let task_desc = job.task_description.clone();
        let agent_cmd_tx = self.agent_cmd_tx.clone();

        // Parse timezone if provided, otherwise use system local
        let tz: Tz = job
            .timezone
            .as_ref()
            .and_then(|s| s.parse::<Tz>().ok())
            .unwrap_or(chrono_tz::UTC);

        let cron_job = Job::new_async_tz(schedule_str.as_str(), tz, move |_uuid, _l| {
            let task_desc = task_desc.clone();
            let agent_cmd_tx = agent_cmd_tx.clone();
            Box::pin(async move {
                info!("Triggering scheduled task: {}", task_desc);
                let cmd = AgentCommand::ChatMessage {
                    content: format!("Proactive Trigger: {}", task_desc),
                    provider: None,
                    model_id: None,
                    agent_id: None,
                };
                if let Err(e) = agent_cmd_tx.send(cmd).await {
                    error!("Failed to send agent command from cron: {}", e);
                }
            })
        })?;

        let scheduler_id = self.scheduler.add(cron_job).await?;

        self.jobs.lock().await.insert(our_id, job);
        self.scheduler_ids.lock().await.insert(our_id, scheduler_id);

        Ok(())
    }

    pub async fn remove_job(&self, id: Uuid) -> Result<()> {
        let mut scheduler_ids = self.scheduler_ids.lock().await;
        if let Some(scheduler_id) = scheduler_ids.remove(&id) {
            self.scheduler.remove(&scheduler_id).await?;
            self.jobs.lock().await.remove(&id);
            self.save_jobs().await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Job not found"))
        }
    }

    pub async fn list_jobs(&self) -> Vec<CronJob> {
        self.jobs.lock().await.values().cloned().collect()
    }
}
