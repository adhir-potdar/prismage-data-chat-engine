#!/bin/bash
###############################################################################
# Create Collection Metadata Table
#
# Creates namespace-specific collection metadata table for fast lookups.
# Table name format: embeddings_collection_metadata_<namespace>
#
# Usage:
#   ./create_collection_metadata_table.sh <namespace> [db_url]
#
# Examples:
#   ./create_collection_metadata_table.sh default
#   ./create_collection_metadata_table.sh revenue_mgmt
#   ./create_collection_metadata_table.sh default "postgresql://user:pass@localhost/dbname"
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print usage
print_usage() {
    echo "Usage: $0 <namespace> [db_url]"
    echo ""
    echo "Arguments:"
    echo "  namespace   Namespace for the table (required)"
    echo "              Examples: default, revenue_mgmt, prod"
    echo ""
    echo "  db_url      PostgreSQL connection URL (optional)"
    echo "              If not provided, reads from DATABASE_URL env variable"
    echo "              Format: postgresql://user:pass@host:port/dbname"
    echo ""
    echo "Examples:"
    echo "  $0 default"
    echo "  $0 revenue_mgmt"
    echo "  $0 default 'postgresql://user:pass@localhost:5432/mydb'"
}

# Check arguments
if [ $# -lt 1 ]; then
    print_error "Missing required argument: namespace"
    echo ""
    print_usage
    exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_usage
    exit 0
fi

NAMESPACE="$1"
DB_URL="${2:-$DATABASE_URL}"

# Validate namespace (alphanumeric and underscore only)
if ! [[ "$NAMESPACE" =~ ^[a-z0-9_]+$ ]]; then
    print_error "Invalid namespace: $NAMESPACE"
    print_info "Namespace must contain only lowercase letters, numbers, and underscores"
    exit 1
fi

# Check if database URL is provided
if [ -z "$DB_URL" ]; then
    print_error "Database URL not provided"
    print_info "Either pass as second argument or set DATABASE_URL environment variable"
    exit 1
fi

# Generate table name
TABLE_NAME="embeddings_collection_metadata_${NAMESPACE}"

print_info "Creating collection metadata table for namespace: ${NAMESPACE}"
print_info "Table name: ${TABLE_NAME}"
echo ""

# Create table SQL
CREATE_TABLE_SQL="
CREATE TABLE IF NOT EXISTS ${TABLE_NAME} (
    collection_name VARCHAR(200) PRIMARY KEY,
    dimension VARCHAR(100) NOT NULL,
    time_granularity VARCHAR(10) NOT NULL,
    dimension_values JSONB,
    period1_start_date INTEGER NOT NULL,
    period1_end_date INTEGER NOT NULL,
    period2_start_date INTEGER NOT NULL,
    period2_end_date INTEGER NOT NULL,
    total_embeddings INTEGER DEFAULT 0,
    last_updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"

# Create indexes SQL
CREATE_INDEXES_SQL="
-- Composite index for dimension + granularity queries
CREATE INDEX IF NOT EXISTS ix_${TABLE_NAME}_dim_gran
    ON ${TABLE_NAME}(dimension, time_granularity);

-- Indexes for period1 date range queries
CREATE INDEX IF NOT EXISTS ix_${TABLE_NAME}_period1_dates
    ON ${TABLE_NAME}(period1_start_date, period1_end_date);

-- Indexes for period2 date range queries
CREATE INDEX IF NOT EXISTS ix_${TABLE_NAME}_period2_dates
    ON ${TABLE_NAME}(period2_start_date, period2_end_date);

-- Individual indexes for fast filtering
CREATE INDEX IF NOT EXISTS ix_${TABLE_NAME}_dimension
    ON ${TABLE_NAME}(dimension);

CREATE INDEX IF NOT EXISTS ix_${TABLE_NAME}_time_gran
    ON ${TABLE_NAME}(time_granularity);
"

# Execute SQL
print_info "Creating table ${TABLE_NAME}..."
if psql "$DB_URL" -c "$CREATE_TABLE_SQL" 2>&1; then
    print_success "Table created successfully"
else
    print_error "Failed to create table"
    print_info "Make sure 'psql' is installed and in your PATH"
    print_info "Alternatively, use: python create_collection_metadata_table.py --namespace ${NAMESPACE}"
    exit 1
fi

print_info "Creating indexes..."
if psql "$DB_URL" -c "$CREATE_INDEXES_SQL" > /dev/null 2>&1; then
    print_warning "Some indexes may have failed (they might already exist)"
fi

# Verify table exists
print_info "Verifying table..."
TABLE_EXISTS=$(psql "$DB_URL" -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '${TABLE_NAME}');")

if [[ "$TABLE_EXISTS" =~ "t" ]]; then
    print_success "Table verified: ${TABLE_NAME}"

    # Show table info
    echo ""
    print_info "Table structure:"
    psql "$DB_URL" -c "\\d ${TABLE_NAME}"

    echo ""
    print_success "✨ Collection metadata table setup complete!"
    print_info "Next steps:"
    echo "  1. Run metadata builder: python build_collection_metadata.py --namespace ${NAMESPACE}"
    echo "  2. Test V4: python ../yield_mgmt_qa_ex_charts_v4.py"
else
    print_error "Table verification failed"
    exit 1
fi
