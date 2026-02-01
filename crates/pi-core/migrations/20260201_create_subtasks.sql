CREATE TABLE IF NOT EXISTS subtasks (
    id TEXT PRIMARY KEY NOT NULL,
    root_task_id TEXT NOT NULL,
    parent_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    result TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY(root_task_id) REFERENCES tasks(id)
);
