create extension if not exists vector;

create table emails (
  id text primary key,
  message_id text unique,
  thread_id text,
  date timestamptz,
  sender_name text,
  sender_address text,
  subject text,
  body_clean text,
  kind text check (kind in ('human','jira'))
);

create table threads (
  thread_id text primary key,
  root_message_id text,
  subject text
);

create table jira_events (
  id serial primary key,
  ticket text,
  action text,
  actor text,
  date timestamptz,
  status text,
  fix_version text,
  title text,
  source_email_id text unique references emails
);

create table documents (
  doc_id text primary key,
  source_kind text,
  source_id text,
  chunk_index int,
  text text,
  token_count int,
  date timestamptz,
  sender text,
  subject text
);

create table llm_cache (
  email_id text,
  prompt_version text,
  model text,
  response jsonb,
  primary key (email_id, prompt_version, model)
);

create table failures (
  id serial primary key,
  stage text,
  item_id text,
  error text,
  at timestamptz default now()
);
