"""
Prompt Architect agent.
Transforms any user input into three production-grade execution prompts.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Literal, List

from pydantic import BaseModel, Field
from mistralai.client import Mistral
from backend.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PromptBlock(BaseModel):
    title: str = Field(..., description="PLAN, BUILD, or OPTIMIZE")
    purpose: str = Field(..., description="One sentence describing what this prompt does")
    content: str = Field(..., description="Full prompt text")
    domain: str = Field(..., description="The detected domain")


class PromptArchitectResponse(BaseModel):
    prompts: List[PromptBlock]
    domain: str
    query_id: str
    response_type: str = 'prompt_architect'

# ---------------------------------------------------------------------------
# Core domain mapping & rules
# ---------------------------------------------------------------------------

DOMAINS = {
    'Full Stack Development': ['full stack', 'fullstack', 'next.js', 'nextjs', 'django', 'rails', 'sveltekit', 'remix', 'mern', 'mean'],
    'Frontend Development': ['frontend', 'front-end', 'css', 'html', 'react', 'vue', 'angular', 'tailwind', 'styled-components', 'sass', 'flexbox', 'grid', 'dom', 'component', 'styling'],
    'Backend Development': ['backend', 'back-end', 'node', 'express', 'flask', 'fastapi', 'spring boot', 'golang', 'go ', 'java', 'nest.js', 'nestjs', 'server'],
    'Mobile Development': ['mobile', 'app', 'ios', 'android', 'swift', 'kotlin', 'react native', 'flutter', 'xcode', 'gradle', 'expo'],
    'AI Engineering': ['llm', 'rag', 'embeddings', 'openai', 'anthropic', 'gpt', 'claude', 'langchain', 'llama', 'prompt', 'vector db', 'pinecone', 'chroma', 'llamaIndex'],
    'Machine Learning': ['machine learning', ' ml ', 'pytorch', 'tensorflow', 'scikit-learn', 'numpy', 'pandas', 'sklearn', 'cnn', 'rnn', 'transformer', 'classification', 'regression', 'dataset'],
    'Agent Systems': ['agent', 'multi-agent', 'crewai', 'langgraph', 'autogen', 'swarm', 'reflection', 'tool calling', 'function calling', 'cognitive architecture'],
    'SaaS Products': ['saas', 'software as a service', 'subscription', 'stripe', 'billing', 'tenant', 'multi-tenant', 'payment gateway', 'churn'],
    'Automation': ['automation', 'scraping', 'scrape', 'selenium', 'puppeteer', 'playwright', 'cron', 'workflow', 'automate', 'webhook'],
    'APIs': ['api', 'rest', 'graphql', 'grpc', 'endpoint', 'swagger', 'openapi', 'postman', 'soap'],
    'Databases': ['database', 'db ', 'sql', 'postgres', 'mysql', 'mongodb', 'redis', 'prisma', 'neo4j', 'sqlite', 'nosql', 'orm'],
    'DevOps': ['devops', 'docker', 'kubernetes', 'k8s', 'ci/cd', 'cicd', 'github actions', 'jenkins', 'nginx', 'prometheus', 'grafana', 'pm2'],
    'Cloud Infrastructure': ['cloud', 'aws', 'azure', 'gcp', 'terraform', 's3', 'ec2', 'lambda', 'serverless', 'cloudfront', 'vpc'],
    'Cybersecurity': ['security', 'auth', 'jwt', 'oauth', 'encryption', 'cors', 'xss', 'csrf', 'ssl', 'tls', 'vulnerability', 'hack', 'penetration'],
    'Data Engineering': ['data engineering', 'spark', 'hadoop', 'kafka', 'etl', 'airflow', 'snowflake', 'bigquery', 'data lake', 'dbt', 'pipeline'],
    'UI/UX Design': ['ui/ux', 'ui/ux design', 'figma', 'wireframe', 'typography', 'color palette', 'contrast', 'usability', 'prototype', 'kerning'],
    'Product Design': ['product design', 'user flow', 'user journey', 'mockup', 'wireframing', 'onboarding', 'roadmap', 'user experience'],
    'Business Strategy': ['business', 'strategy', 'monetization', 'pitch deck', 'competitor', 'pricing', 'roi', 'cac', 'ltv', 'market fit'],
    'Research': ['research', 'academic', 'literature', 'hypothesis', 'citation', 'latex', 'scientific', 'experiment', 'thesis'],
    'Content Systems': ['cms', 'markdown', 'headless', 'blog', 'seo', 'editor', 'rich-text', 'wordpress', 'ghost']
}

# Coding-specific keywords that always route to PROMPT_ARCHITECT
_CODE_TERMS = frozenset({
    'python', 'javascript', 'typescript', 'java', 'golang', 'rust', 'kotlin',
    'swift', 'code', 'class', 'function', 'algorithm', 'sort', 'loop',
    'recursion', 'async', 'api', 'endpoint', 'database', 'query', 'schema',
    'component', 'hook', 'module', 'script', 'implement', 'refactor',
})

# Phrases that always route to PROMPT_ARCHITECT regardless of question marks
_ARCHITECT_PHRASES = [
    r'give me.{0,20}code',
    r'write.{0,20}code',
    r'show me.{0,20}code',
    r'how do i (build|create|implement|make)',
    r'i want to (build|create|make|develop)',
    r'help me (build|create|design|implement)',
    r'(build|create|design|make|implement|develop|generate|scaffold|architect|plan|optimize|refactor)\s+\w',
]

# Keywords always routing to PROMPT_ARCHITECT
_ARCHITECT_KEYWORDS = frozenset({
    'saas', 'startup', 'microservice', 'dashboard', 'boilerplate',
    'deployment', 'docker', 'kubernetes', 'ci/cd', 'pipeline', 'devops',
    'landing page', 'mobile app', 'tech stack', 'system design',
    'api design', 'database schema', 'architecture', 'scaffold',
})

# Paper query phrases — always route to PAPER_QUERY
_PAPER_PHRASES = [
    r'what (is|are|was|were) the (main|key|primary|core)',
    r'(summarize|explain|describe|what does|tell me about)',
    r'(contribution|methodology|result|finding|limitation|dataset|experiment|author|abstract)',
    r'according to (the paper|this paper|the study|the research)',
    r'(paper|study|research|authors?|article) (say|mention|discuss|show|find|conclude)',
]

def classify_intent(query: str) -> str:
    q_lower = query.lower().strip()
    
    # 1. Check paper-specific phrases first (highest priority)
    for phrase in _PAPER_PHRASES:
        if re.search(phrase, q_lower):
            return 'PAPER_QUERY'
    
    # 2. Check architect phrases (catches "give me python code" regardless of ?)
    for phrase in _ARCHITECT_PHRASES:
        if re.search(phrase, q_lower):
            return 'PROMPT_ARCHITECT'
    
    # 3. Check code terms — if query contains coding keywords, route to architect
    tokens = set(re.sub(r'[^\w\s]', ' ', q_lower).split())
    if tokens & _CODE_TERMS:
        return 'PROMPT_ARCHITECT'
    
    # 4. Check architect keywords
    for kw in _ARCHITECT_KEYWORDS:
        if kw in q_lower:
            return 'PROMPT_ARCHITECT'
    
    # 5. No question mark AND more than 8 words → likely a command
    words = q_lower.split()
    if '?' not in query and len(words) > 8:
        return 'PROMPT_ARCHITECT'
    
    # Default: treat as paper query
    return 'PAPER_QUERY'


def detect_domain(query: str) -> str:
    """Detects domain classification for the query based on keyword overlap."""
    clean_query = query.lower()
    best_domain = "Full Stack Development"
    max_score = 0
    
    for domain, keywords in DOMAINS.items():
        score = 0
        for kw in keywords:
            score += clean_query.count(kw)
        if score > max_score:
            max_score = score
            best_domain = domain
            
    return best_domain


_PROMPT_HEADER_RE = re.compile(
    r'(?:#{1,3}\s*)?(?:\*{1,2})?PROMPT\s*([123])(?:\*{1,2})?(?:\s*[-:—])?',
    re.IGNORECASE,
)

_PROMPT_TITLES = {1: 'PLAN', 2: 'BUILD', 3: 'OPTIMIZE'}
_PROMPT_PURPOSES = {
    1: 'Complete implementation blueprint — requirements, architecture, tech stack, database, API, folder structure, security, scalability.',
    2: 'Full implementation instructions — frontend, backend, database, APIs, integrations, error handling, validation, best practices.',
    3: 'Audit and harden — code review, security, bugs, edge cases, performance, testing, deployment readiness, documentation.',
}

def _parse_prompt_blocks(raw: str, domain: str) -> list[PromptBlock]:
    """Split Mistral response into exactly 3 PromptBlock objects."""
    matches = list(_PROMPT_HEADER_RE.finditer(raw))
    
    blocks: list[PromptBlock] = []
    
    if len(matches) < 3:
        # Fallback: split raw into thirds if headers not found
        logger.warning("Prompt headers not detected — falling back to equal split")
        chunk_size = max(1, len(raw) // 3)
        parts = [raw[i:i+chunk_size] for i in range(0, len(raw), chunk_size)]
        parts = (parts + ['', '', ''])[:3]
        for i, content in enumerate(parts, start=1):
            blocks.append(PromptBlock(
                title=_PROMPT_TITLES[i],
                purpose=_PROMPT_PURPOSES[i],
                content=content.strip(),
                domain=domain,
            ))
        return blocks
    
    # Extract content between each header
    for idx, match in enumerate(matches[:3]):
        num = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        
        # Clean any leading separator character or empty lines
        lines = content.splitlines()
        while lines and (lines[0].strip() == "---" or not lines[0].strip()):
            lines = lines[1:]
        while lines and (lines[-1].strip() == "---" or not lines[-1].strip()):
            lines = lines[:-1]
        cleaned_content = "\n".join(lines).strip()
        
        blocks.append(PromptBlock(
            title=_PROMPT_TITLES.get(num, f'PROMPT {num}'),
            purpose=_PROMPT_PURPOSES.get(num, ''),
            content=cleaned_content,
            domain=domain,
        ))
    
    return blocks


_FALLBACK_PROMPTS = [
    "You are a Principal Solutions Architect. Produce a complete implementation blueprint for the following request. Include: requirements analysis, system architecture, tech stack with justification, database schema, API design, folder structure, security model, scalability plan, and development roadmap. Be precise and thorough. Request: {query}",
    "You are a Principal Software Engineer. Implement the following completely. Include: full frontend implementation, backend services, database setup, API endpoints with validation and error handling, authentication, integrations, and deployment configuration. Follow production best practices. No architecture debates. Request: {query}",
    "You are a Principal QA and Security Engineer. Audit the implementation of the following completely. Include: code review, security vulnerabilities, bug detection, edge cases, performance bottlenecks, test suite generation, deployment readiness checklist, and documentation gaps. Request: {query}",
]

async def generate_prompt_architect_response(
    query: str,
    settings: Settings,
) -> PromptArchitectResponse:
    domain = detect_domain(query)
    query_id = str(uuid.uuid4())
    
    system_prompt = """You are an Elite Prompt Architect and Principal Software Engineer.
Your sole purpose is to transform the user's input into exactly THREE optimized, production-grade prompts for use with AI coding agents.

RULES:
- Output EXACTLY three prompts. No more. No less.
- Each prompt starts with its header on its own line: PROMPT 1, PROMPT 2, or PROMPT 3
- PROMPT 1 focuses on PLANNING and ARCHITECTURE only. No code.
- PROMPT 2 focuses on IMPLEMENTATION only. No architecture debates.
- PROMPT 3 focuses on REVIEW, TESTING, and OPTIMIZATION only.
- Infer missing technical context. Never ask follow-up questions.
- Include relevant technical concerns automatically: auth, security, scalability, error handling, validation, testing, deployment.
- Assume production-level quality always.
- No fluff. No repetition. No generic advice. High information density.
- Never explain your reasoning. Never add summaries. Never add introductions.
- Never use phrases like "here is your prompt" or "optimized prompt".
- Each prompt must be self-contained and immediately usable by a coding agent."""

    try:
        client = Mistral(api_key=settings.mistral_api_key)
        response = await asyncio.wait_for(
            client.chat.complete_async(
                model=settings.mistral_large_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query},
                ],
                temperature=0.3,
                max_tokens=4096,
            ),
            timeout=60.0,
        )
        raw = response.choices[0].message.content or ''
        blocks = _parse_prompt_blocks(raw, domain)
    
    except asyncio.TimeoutError:
        logger.warning("Prompt Architect: Mistral timeout — using local fallback templates")
        blocks = [
            PromptBlock(
                title=_PROMPT_TITLES[i],
                purpose=_PROMPT_PURPOSES[i],
                content=template.format(query=query),
                domain=domain,
            )
            for i, template in enumerate(_FALLBACK_PROMPTS, start=1)
        ]
    
    except Exception as exc:
        logger.error("Prompt Architect: Mistral error: %s", exc)
        blocks = [
            PromptBlock(
                title=_PROMPT_TITLES[i],
                purpose=_PROMPT_PURPOSES[i],
                content=template.format(query=query),
                domain=domain,
            )
            for i, template in enumerate(_FALLBACK_PROMPTS, start=1)
        ]
    
    return PromptArchitectResponse(
        prompts=blocks,
        domain=domain,
        query_id=query_id,
        response_type='prompt_architect',
    )
