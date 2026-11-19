"""Keep the accepted OTel parent for the lifetime of a message."""

import sqlalchemy as sa
from alembic import op

revision = "9dcff89b3417"
down_revision = "c3f1a9b2e6d4"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("messages", sa.Column("otel_parent_context", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("messages", "otel_parent_context")
