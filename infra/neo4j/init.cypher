// =============================================================================
//  init.cypher — Neo4j Schema Initialization for GraphRAG Research Assistant
//
//  Run this script ONCE before the first application launch:
//
//    Option A (docker exec):
//      docker exec -it graphrag-neo4j \
//        cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
//        --file /var/lib/neo4j/import/init.cypher
//
//    Option B (Python):
//      python -m backend.infra.schema_init
//
//  All statements are idempotent (IF NOT EXISTS) — safe to re-run.
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// UNIQUENESS CONSTRAINTS
// Each constraint also creates a backing b-tree index automatically.
// ─────────────────────────────────────────────────────────────────────────────

// Paper node: unique on paper_id
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;

// Entity node: unique on (entity_id) — scoped globally
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

// Chunk node: unique on chunk_id
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

// Job node: unique on job_id (used for tracking ingestion pipeline state)
CREATE CONSTRAINT job_id_unique IF NOT EXISTS
FOR (j:Job) REQUIRE j.job_id IS UNIQUE;

// ─────────────────────────────────────────────────────────────────────────────
// RANGE (B-TREE) INDEXES
// For efficient equality and range lookups on frequently queried properties.
// ─────────────────────────────────────────────────────────────────────────────

// Paper — look up by domain, primary_domain, created_at
CREATE INDEX paper_domain_idx IF NOT EXISTS
FOR (p:Paper) ON (p.primary_domain);

CREATE INDEX paper_created_at_idx IF NOT EXISTS
FOR (p:Paper) ON (p.created_at);

CREATE INDEX paper_is_interdisciplinary_idx IF NOT EXISTS
FOR (p:Paper) ON (p.is_interdisciplinary);

// Entity — look up by label (type) and by paper_id for scoped queries
CREATE INDEX entity_label_idx IF NOT EXISTS
FOR (e:Entity) ON (e.label);

CREATE INDEX entity_paper_id_idx IF NOT EXISTS
FOR (e:Entity) ON (e.paper_id);

// Chunk — look up by paper_id and section for retrieval
CREATE INDEX chunk_paper_id_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.paper_id);

CREATE INDEX chunk_section_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.section);

CREATE INDEX chunk_chunk_index_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.chunk_index);

// Job — look up by status for queue management
CREATE INDEX job_status_idx IF NOT EXISTS
FOR (j:Job) ON (j.status);

CREATE INDEX job_paper_id_idx IF NOT EXISTS
FOR (j:Job) ON (j.paper_id);

// ─────────────────────────────────────────────────────────────────────────────
// FULL-TEXT INDEXES
// Used by GraphAgent for keyword / semantic entity search.
// ─────────────────────────────────────────────────────────────────────────────

// Full-text search across Entity name and description
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.description]
OPTIONS {
  indexConfig: {
    `fulltext.analyzer`: 'english',
    `fulltext.eventually_consistent`: false
  }
};

// Full-text search across Paper title and abstract
CREATE FULLTEXT INDEX paper_fulltext IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title, p.abstract]
OPTIONS {
  indexConfig: {
    `fulltext.analyzer`: 'english',
    `fulltext.eventually_consistent`: false
  }
};

// Full-text search across Chunk text (for direct text retrieval from graph)
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text]
OPTIONS {
  indexConfig: {
    `fulltext.analyzer`: 'english',
    `fulltext.eventually_consistent`: false
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// SAMPLE VERIFICATION QUERY
// Un-comment and run interactively to confirm all constraints/indexes exist:
//
//   SHOW CONSTRAINTS;
//   SHOW INDEXES;
// ─────────────────────────────────────────────────────────────────────────────

// Verification: count schema objects (should be > 0 after init)
RETURN
  'Schema initialization complete' AS message,
  datetime() AS initialized_at;
