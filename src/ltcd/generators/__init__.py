from contextlib import suppress

from .string_generators import (
    generate_random_data,
    generate_random_egnlish_row,
    get_current_date,
)

with suppress(ImportError):
    from .doc_generator import (
        DocumentConfig,
        DocumentField,
        DocumentGenerator,
        Table,
    )
