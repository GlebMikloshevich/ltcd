import random
import string

from datetime import (
    datetime,
    timedelta,
)


def generate_random_data(format: str="%d.%m.%Y", min_year: int=1900, max_year: int=2030) -> str:
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    date =  start + (end - start) * random.random()  # noqa: S311
    return date.strftime(format=format)


def get_current_date(format: str="%d.%m.%Y") -> str:
    current_datetime = datetime.now()
    return current_datetime.strftime(format=format)


def generate_random_egnlish_row(
    min_length: int = 1,
    max_length: int = 10,
    alphabet: str = string.ascii_letters + string.digits + string.whitespace,
) -> str:
    length = random.randint(min_length, max_length)  # noqa: S311
    return "".join(random.choice(alphabet) for _ in range(length))  # noqa: S311


class RandomStringGenerator:

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 15,
        charset: str = "alphanumeric",
    ):
        self.min_length = min_length
        self.max_length = max_length

        if charset == "alphanumeric":
            self.alphabet = string.ascii_letters + string.digits
        elif charset == "alpha":
            self.alphabet = string.ascii_letters
        elif charset == "numeric":
            self.alphabet = string.digits
        elif charset == "alphanum_space":
            self.alphabet = string.ascii_letters + string.digits + " "
        else:
            self.alphabet = charset

    def generate(self) -> str:
        length = random.randint(self.min_length, self.max_length)  # noqa: S311
        return "".join(random.choice(self.alphabet) for _ in range(length))  # noqa: S311


class DateStringGenerator:

    def __init__(
        self,
        format: str = "%d.%m.%Y",
        min_year: int = 1900,
        max_year: int = 2030,
    ):
        self.format = format
        self.min_year = min_year
        self.max_year = max_year

    def generate(self) -> str:
        return generate_random_data(self.format, self.min_year, self.max_year)


class PatternStringGenerator:

    def __init__(self, pattern: str = "ID-####"):
        self.pattern = pattern

    def generate(self) -> str:
        result = []
        for char in self.pattern:
            if char == '#':
                result.append(random.choice(string.digits))  # noqa: S311
            elif char == '@':
                result.append(random.choice(string.ascii_uppercase))  # noqa: S311
            elif char == '?':
                result.append(random.choice(string.ascii_lowercase))  # noqa: S311
            elif char == '*':
                result.append(random.choice(string.ascii_letters + string.digits))  # noqa: S311
            else:
                result.append(char)
        return "".join(result)


def get_string_generator(generator_type: str = "random", **kwargs):
    """
    Factory function to create string generators.

    Args:
        generator_type: Type of generator ("random", "date", "pattern")
        **kwargs: Arguments passed to generator constructor

    Returns:
        String generator instance
    """
    if generator_type == "random":
        return RandomStringGenerator(**kwargs)
    elif generator_type == "date":
        return DateStringGenerator(**kwargs)
    elif generator_type == "pattern":
        return PatternStringGenerator(**kwargs)
    else:
        return RandomStringGenerator(**kwargs)
