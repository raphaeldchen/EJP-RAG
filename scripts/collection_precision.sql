-- Per-collection precision from audit_feedback.
-- Run in the Supabase SQL editor after lawyer sessions.
--
-- "Precision" here = fraction of shown chunks labeled BINDING or RELEVANT.
-- A collection with high precision but low representation rate is underweighted;
-- one with low precision but high representation is overweighted.

with totals as (
    select
        collection_id,
        count(*)                                                            as shown,
        count(*) filter (where label in ('BINDING', 'RELEVANT'))           as relevant,
        count(*) filter (where label = 'BINDING')                          as binding_only,
        count(*) filter (where label = 'IRRELEVANT')                       as irrelevant,
        round(avg(rrf_score)::numeric, 5)                                  as avg_rrf_score,
        round(avg(ce_score)::numeric,  3)                                  as avg_ce_score
    from audit_feedback
    group by collection_id
),
grand as (
    select sum(shown) as total_shown from totals
)
select
    t.collection_id,
    t.shown,
    round(t.shown::numeric / g.total_shown * 100, 1)                      as pct_of_pool,
    t.relevant,
    t.irrelevant,
    round(t.relevant::numeric / nullif(t.shown, 0) * 100, 1)              as precision_pct,
    t.binding_only,
    t.avg_rrf_score,
    t.avg_ce_score
from totals t, grand g
order by precision_pct desc nulls last;
