# Skill-Memory

Skill-Memory is a compact memory plugin for agents and subagents. It captures behavioral signals from conversations, organizes them in a DAG, and feeds that context back into later turns so the agent can respond with more continuity and better judgment.

It is built for teams that want practical leverage from small, focused tools instead of another general-purpose framework. The value is in keeping the scope tight, the dependencies light, and the integration simple enough to fit real workflows.

This is especially useful for subagents with clearly defined tasks inside complex company processes. In that setting, knowing how to act cannot be reduced to a system prompt. Giving the subagent access to this plugin makes it more aware of the process it is operating in and more effective inside that context.

## How to run

The project works through both MCP and A2A.

### MCP

1. Configure the environment in `.env`.
2. Start the stack:
   ```bash
   docker compose up
   ```
3. Connect with the MCP client entrypoint.

### A2A

1. Configure the environment in `.env`.
2. Start the stack:
   ```bash
   docker compose up
   ```
3. Connect with the A2A client.

## Notes

- The codebase stays intentionally small, which makes it easier to adopt, inspect, and extend without a heavy platform effort.
- Dependencies are few and replaceable, so the project remains lightweight and avoids unnecessary lock-in.
- The DAG memory can grow incrementally as new behavior patterns emerge, without turning the project into yet another heavyweight framework.
