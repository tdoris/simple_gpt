# CLAUDE.md - Guidance for AI Assistants

## Build & Test Commands
- Build: `npm run build`
- Lint: `npm run lint`
- Typecheck: `npm run typecheck` 
- Test all: `npm test`
- Test single: `npm test -- -t "test name"`
- Dev server: `npm run dev`

## Model Usage Guidelines
- **LOCAL MODELS ONLY**: Focus on training and running models with our custom implementation
- Do not suggest or rely on external APIs for text generation
- For now, avoid GPT-2 implementations, as they are not working correctly with our codebase
- Focus on custom models trained within our system

## Code Style Guidelines
- **Format**: Use Prettier with default settings
- **Imports**: Group and sort imports: built-ins, external, internal
- **Types**: Use TypeScript. Prefer explicit types over `any`
- **Naming**: camelCase for variables/functions, PascalCase for classes/components
- **Error Handling**: Use try/catch with proper error typing
- **Documentation**: JSDoc for public APIs
- **Component Structure**: Functional components with hooks
- **State Management**: Prefer React Context for global state
- **Testing**: Jest for unit tests, React Testing Library for components

## Git Guidelines
- **IMPORTANT**: NEVER create a git commit unless explicitly requested by the user
- Prepare changes and show diffs when requested, but always wait for user confirmation before committing
- When asked to make changes, stage the relevant files but don't commit them unless specifically instructed

## Scripting Guidelines
- **DO NOT** create scripts that use screen, tmux, or other session managers
- Training runs in a terminal in a virtual desktop, so disconnects are not a concern
- Avoid background processes in scripts (no `&` at end of commands)
- For monitoring, create separate monitor scripts rather than background processes

## Testing Guidelines
- **ALWAYS** test new code before declaring it complete
- Run basic tests with simplified inputs to verify functionality
- Check for edge cases and potential errors
- Validate outputs against expected behavior
- For model-related code, test with small models or sample data first

This document will be updated as the project evolves.