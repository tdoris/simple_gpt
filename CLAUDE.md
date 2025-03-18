# CLAUDE.md - Guidance for AI Assistants

## Build & Test Commands
- Build: `npm run build`
- Lint: `npm run lint`
- Typecheck: `npm run typecheck` 
- Test all: `npm test`
- Test single: `npm test -- -t "test name"`
- Dev server: `npm run dev`

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

This document will be updated as the project evolves.