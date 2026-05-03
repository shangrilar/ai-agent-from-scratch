"""Skill discovery and management for code execution agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillInfo:
    """Information about a discovered skill."""
    name: str
    description: str
    path: Path


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown file."""
    pattern = r'^---\s*\n(.*?)\n---'
    match = re.match(pattern, content, re.DOTALL)
    if not match:
        return {}
    result = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


def load_skill(skill_dir: Path) -> SkillInfo | None:
    """Load skill info from a directory.

    Looks for a SKILL.md file with metadata about the skill.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding='utf-8')
    frontmatter = parse_frontmatter(content)

    name = frontmatter.get('name')
    description = frontmatter.get('description')
    if not name or not description:
        return None

    return SkillInfo(name=name, description=description, path=skill_dir)


def discover_skills(skills_path: str | Path) -> list[SkillInfo]:
    """Discover all skills in a directory.

    Each subdirectory is treated as a potential skill.
    """
    skills_dir = Path(skills_path)
    if not skills_dir.exists():
        return []

    skills = []
    for item in sorted(skills_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            skill = load_skill(item)
            if skill:
                skills.append(skill)
    return skills


def generate_skills_prompt(
    skills: list[SkillInfo],
    sandbox_path: str = "/home/user/skills",
) -> str:
    """Generate a prompt describing available skills.

    This prompt is added to the agent's instructions to inform it
    about available skills that can be used in code execution.
    """
    if not skills:
        return ""

    lines = [
        "## Available Skills",
        "The following skills are available in the sandbox environment:",
        "",
    ]

    for skill in skills:
        lines.append(f"### {skill.name}")
        lines.append(f"- Description: {skill.description}")
        lines.append(f"- Path: {sandbox_path}/{skill.name}/")
        lines.append(f"- Read the SKILL.md for usage instructions: {sandbox_path}/{skill.name}/SKILL.md")
        lines.append("")

    lines.append(
        "You can import and use these skills in your Python code. "
        "Read the SKILL.md file first to understand how to use each skill."
    )

    return "\n".join(lines)
