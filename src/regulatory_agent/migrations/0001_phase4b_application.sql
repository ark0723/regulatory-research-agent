CREATE TABLE IF NOT EXISTS regulatory_app.conversations (
    conversation_id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS regulatory_app.runs (
    run_id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL
        REFERENCES regulatory_app.conversations(conversation_id),
    thread_id text NOT NULL UNIQUE,
    question text NOT NULL,
    status text NOT NULL CHECK (status IN (
        'created', 'running', 'awaiting_human_review',
        'completed', 'closed_for_edit', 'failed'
    )),
    termination_reason text,
    latest_answer text NOT NULL DEFAULT '',
    citations jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_count integer NOT NULL DEFAULT 0 CHECK (source_count >= 0),
    source_types jsonb NOT NULL DEFAULT '[]'::jsonb,
    evaluation_decision text,
    evaluation_total_score integer,
    evaluation_error text,
    validation_is_valid boolean,
    evaluation_retry_count integer NOT NULL DEFAULT 0,
    revision_count integer NOT NULL DEFAULT 0,
    research_retry_count integer NOT NULL DEFAULT 0,
    pending_review jsonb,
    review_claim jsonb,
    error text,
    version bigint NOT NULL DEFAULT 0 CHECK (version >= 0),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    completed_at timestamptz
);

CREATE INDEX IF NOT EXISTS runs_conversation_created_idx
    ON regulatory_app.runs (conversation_id, created_at, run_id);
CREATE INDEX IF NOT EXISTS runs_status_updated_idx
    ON regulatory_app.runs (status, updated_at);

CREATE TABLE IF NOT EXISTS regulatory_app.conversation_messages (
    message_id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL
        REFERENCES regulatory_app.conversations(conversation_id),
    run_id uuid NOT NULL REFERENCES regulatory_app.runs(run_id),
    sequence bigint NOT NULL CHECK (sequence >= 1),
    role text NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    kind text NOT NULL CHECK (kind IN (
        'question', 'answer', 'human_review_request',
        'human_feedback', 'run_status'
    )),
    content text NOT NULL,
    citations jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL,
    UNIQUE (conversation_id, sequence)
);

CREATE INDEX IF NOT EXISTS messages_run_sequence_idx
    ON regulatory_app.conversation_messages (run_id, sequence);

CREATE TABLE IF NOT EXISTS regulatory_app.human_review_actions (
    review_action_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES regulatory_app.runs(run_id),
    review_request_id uuid NOT NULL,
    action text NOT NULL CHECK (action IN (
        'approve', 'revise', 'research_again', 'edit_question'
    )),
    feedback text NOT NULL,
    idempotency_key text NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE (run_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS review_actions_request_idx
    ON regulatory_app.human_review_actions (run_id, review_request_id);

CREATE TABLE IF NOT EXISTS regulatory_app.review_submission_receipts (
    run_id uuid NOT NULL REFERENCES regulatory_app.runs(run_id),
    idempotency_key text NOT NULL,
    review_request_id uuid NOT NULL,
    action text NOT NULL CHECK (action IN (
        'approve', 'revise', 'research_again', 'edit_question'
    )),
    feedback text NOT NULL,
    result jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS review_receipts_request_idx
    ON regulatory_app.review_submission_receipts (run_id, review_request_id);
