CREATE TABLE IF NOT EXISTS lawyer_accounts (
    id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    email         text UNIQUE NOT NULL,
    password_hash text NOT NULL,
    approved      boolean DEFAULT false,
    created_at    timestamptz DEFAULT now(),
    approved_at   timestamptz
);

CREATE INDEX IF NOT EXISTS lawyer_accounts_email_idx ON lawyer_accounts (email);
CREATE INDEX IF NOT EXISTS lawyer_accounts_approved_idx ON lawyer_accounts (approved);
