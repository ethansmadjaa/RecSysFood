# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RecSysFood is a full-stack food recommendation system with:
- **Backend**: FastAPI (Python 3.13.3+) managed with `uv`
- **Frontend**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS v4 with shadcn/ui components (new-york style)

## Development Commands

### Backend (Python/FastAPI)
Backend uses `uv` for dependency management. From project root:

```bash
# Run the FastAPI server
uv run fastapi dev main.py

# Install dependencies
uv add <package-name>
```

### Frontend (React/Vite)
Frontend uses `pnpm` as the package manager. Navigate to `frontend/` directory:

```bash
cd frontend

# Development server with HMR
pnpm dev

# Build for production
pnpm build

# Lint code
pnpm lint

# Preview production build
pnpm preview

# Install dependencies
pnpm install

# Add shadcn/ui components
pnpx shadcn@latest add <component-name>
```

## Architecture

### Full-Stack Structure
- **Backend**: `/main.py` - FastAPI app entry point, `/routes/` - API routes
- **Frontend**: `/frontend/` - Complete React application

### Frontend Architecture

**Build System**: Vite with React plugin and React Compiler (experimental)

**Styling System**:
- **Tailwind CSS v4** with `@theme inline` syntax in `src/index.css`
- **NO separate `tailwind.config.js`** - all configuration is inline in CSS
- CSS variables system for theming (`:root` for light, `.dark` for dark mode)
- OKLCH color space for all theme colors

**Component System**:
- **shadcn/ui** components in `src/components/ui/`
- Uses Radix UI primitives with custom styling
- Components reference theme colors via CSS variables (e.g., `bg-primary`, `text-foreground`)
- Path alias: `@/` maps to `src/`

**Color Palette**:
The app uses a custom brand color palette defined in `src/index.css`:
- **Soft Linen** (#EDEBE5) - Primary background
- **Sky Haze** (#DBE4EA) - Emphasis & separation boxes
- **Pulse Blue** (#3568F6) - Brand & action color
- **Light Beige** (#F4F2EF) - Secondary background variation
- **White** (#FFFFFF) - Contrast color
- **Dark Blue** (#0B355F) - Headlines & trust elements
- **Dark Black** (#111011) - Body text

All colors are defined in OKLCH format for better perceptual uniformity. When updating colors, ensure they remain in OKLCH format and update both `:root` and `.dark` sections.

**Key Configuration Files**:
- `vite.config.ts`: Vite configuration with path aliases
- `components.json`: shadcn/ui configuration (style: "new-york", cssVariables: true)
- `src/index.css`: Complete theme system including colors, radius, and Tailwind v4 theme mapping
- `tsconfig.json`: TypeScript path mapping for `@/*` alias

### Backend Architecture

**Framework**: FastAPI with minimal setup
- Single entry point in `main.py`
- Routes directory structure at `/routes/`
- Python 3.13.3+ required
- Dependencies managed via `pyproject.toml` with `uv`

## Important Notes

- Frontend uses `pnpm` (version 10.20.0), not npm or yarn
- Backend uses `uv` for package management, not pip
- Path imports use `@/` prefix for frontend source files
- Theme colors MUST be in OKLCH format, not HEX or RGB
- All shadcn/ui components automatically use the CSS variable theme system
