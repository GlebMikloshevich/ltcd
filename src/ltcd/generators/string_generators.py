import random
import string

from datetime import (
    datetime,
    timedelta,
)

def generate_random_data(format: str="%Y-%m-%d", min_year: int=1900, max_year: int=2030) -> str:
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    date =  start + (end - start) * random.random()  # noqa: S311
    return date.strftime(format=format)


def get_current_date(format: str="%Y-%m-%d") -> str:
    current_datetime = datetime.now()
    return current_datetime.strftime(format=format)

def generate_random_egnlish_row(
    min_length: int = 1,
    max_length: int = 10,
    alphabet: str = string.ascii_letters + string.digits + string.whitespace,
) -> str:
    length = random.randint(min_length, max_length)  # noqa: S311
    return "".join(random.choice(alphabet) for _ in range(length))  # noqa: S311
