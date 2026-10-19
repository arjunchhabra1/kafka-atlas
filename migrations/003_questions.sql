create table questions (
  id text primary key,
  type text not null check (type in ('single_hop','inference','comparison','temporal','null')),
  question text not null,
  answer text not null,
  gold_doc_ids text[] not null default '{}',
  status text not null check (status in ('draft','kept','edited','dropped')),
  split text,
  draft_model text,
  prompt_version text,
  reviewed_at timestamptz
);
