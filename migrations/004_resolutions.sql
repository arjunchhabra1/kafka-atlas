create table resolutions (
  resolver text not null,
  mention_id text not null,
  entity_id text not null,
  canonical_name text not null,
  entity_type text not null,
  primary key (resolver, mention_id)
);

create index resolutions_entity_idx on resolutions (resolver, entity_id);
