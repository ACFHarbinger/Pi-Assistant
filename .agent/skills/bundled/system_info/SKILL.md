---
name: System Information
description: Get system information like disk usage, memory, CPU, and running processes
version: 1.0.0
---

# System Information Skill

Provides commands to query system state.

## Usage

When the user asks about system resources, use the shell tool with these commands:

### Disk Usage
```bash
df -h
```

### Memory Usage
```bash
free -h
```

### CPU Information
```bash
lscpu | head -20
```

### Running Processes
```bash
ps aux --sort=-%mem | head -15
```

### System Uptime
```bash
uptime
```

## Examples

- "How much disk space do I have?"
- "What's my memory usage?"
- "Show me the top processes"
- "How long has this system been running?"
