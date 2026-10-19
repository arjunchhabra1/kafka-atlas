create table mentions (
  mention_id text primary key,
  surface text,
  type text,
  email_id text,
  context text,
  source text not null check (source in ('llm','regex','header','jira'))
);

create index mentions_email_id_idx on mentions (email_id);

create table relations (
  id serial primary key,
  type text,
  source_mention_id text,
  target_mention_id text,
  vote_value int,
  binding bool,
  confidence real,
  email_id text
);
