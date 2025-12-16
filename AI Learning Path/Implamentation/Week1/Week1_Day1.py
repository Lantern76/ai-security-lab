password = "pass"
failures = []

# Check 1: Length (minimum 8 characters)
if len(password) < 8:
    failures.append("at least 8 characters")

# Check 2: At least one uppercase letter
if not any(c.isupper() for c in password):
    failures.append("one uppercase letter")

# Check 3: At least one lowercase letter
if not any(c.islower() for c in password):
    failures.append("one lowercase letter")

# Check 4: At least one digit
if not any(c.isdigit() for c in password):
    failures.append("one digit")

# Check 5: At least one symbol (non-alphanumeric)
if all(c.isalnum() for c in password):
    failures.append("one symbol (!@#$%^&* etc.)")

# Report results
if failures:
    print("Password rejected. Missing:", ", ".join(failures) + ".")
else:
    print("Password accepted")

password = "pass"


password = "pass"


def validate_password(password):
    failures = []
    if len(password) < 8:
        failures.append("at least 8 characters")
    if not any(c.isupper() for c in password):
        failures.append("one uppercase letter")
    if not any(c.islower() for c in password):
        failures.append("one lowercase letter")
    if not any(c.isdigit() for c in password):
        failures.append("one digit")
    if all(c.isalnum() for c in password):
        failures.append("one symbol((!@#$%^&* etc.)")

    return failures
