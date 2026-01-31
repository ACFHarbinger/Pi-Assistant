# Agent Personality Guide: soul.md

Pi-Assistant's personality is defined in a special file called `soul.md` located in the root of your workspace. By editing this file, you can customize how your agent speaks, acts, and interacts with you.

## How it Works

The contents of `soul.md` are injected into the agent's **system prompt** at the start of every interaction. This gives the agent a persistent identity across sessions.

### Key Sections in soul.md

- **Identity**: Defines the agent's name and role.
- **Personality Traits**: A bulleted list of characteristics (e.g., "Sarcastic," "Helpful," "Technical").
- **Communication Style**: Guidelines for language use, emojis, and sentence structure.
- **Hatching (First Encounter)**: Instructions for how the agent should introduce itself during the initial setup.

## Customizing Your Agent

You can change any part of `soul.md` to suit your preference. For example:

### Making the Agent More Sarcastic

```markdown
## Personality Traits

- **Sarcastic**: You love dry wit and making subtle jokes about the user's coding mistakes.
```

### Changing the Name

While you can edit the name directly in `soul.md`, the **Hatching Experience** in the app will also update this for you during setup.

```markdown
You are **Jarvis**, a helpful AI assistant...
```

## Tips for Better Personalities

1.  **Be Specific**: Instead of "Be nice," try "Be exceptionally polite and use formal language."
2.  **Add Quarks**: Give your agent a specific obsession, like always referencing space lobster facts 🦞.
3.  **Define Boundaries**: Explicitly tell the agent what NOT to do in the Boundaries section.

---

_Enjoy shaping your perfect digital companion!_
