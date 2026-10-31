"""add_source_to_resource_metrics

Revision ID: fc90c3d3166d
Revises: f6a7b8c9d0e1
Create Date: 2026-10-29 19:50:52.434376

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fc90c3d3166d'
down_revision: Union[str, None] = 'f6a7b8c9d0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add source column with default 'mock'."""
    op.add_column(
        'resource_metrics',
        sa.Column('source', sa.String(length=50), server_default='mock', nullable=False)
    )
    op.create_index(op.f('ix_resource_metrics_source'), 'resource_metrics', ['source'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_resource_metrics_source'), table_name='resource_metrics')
    op.drop_column('resource_metrics', 'source')

