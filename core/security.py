# core/security.py
# All security validation in one place
# Defense in depth — multiple layers of checks

import re
from config import (
    MAX_FILE_SIZE_BYTES,
    ALLOWED_FILE_TYPES,
    MAX_QUESTION_LENGTH,
    DANGEROUS_PATTERNS,
)
from utils.logger import logger


class SecurityError(Exception):
    """Raised when input fails security validation."""
    pass


class SecurityValidator:
    """
    Validates all user input before it enters the system.
    Single Responsibility: ONLY does security checks.
    """

    @staticmethod
    def validate_file_type(file) -> None:
        """
        Check file extension is allowed.
        Raises SecurityError if not allowed.
        """
        # Get extension from filename
        filename = file.name.lower()
        extension = filename.split(".")[-1]  # "pci_dss.PDF" → "pdf"

        if extension not in ALLOWED_FILE_TYPES:
            logger.warning(
                "invalid_file_type",
                filename=file.name,
                extension=extension,
                allowed=ALLOWED_FILE_TYPES,
            )
            raise SecurityError(
                f"File type '.{extension}' is not allowed. "
                f"Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
            )

    @staticmethod
    def validate_file_size(file) -> None:
        """
        Check file is under size limit.
        Raises SecurityError if too large.
        """
        # file.size is in bytes
        size_mb = file.size / (1024 * 1024)

        if file.size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                "file_too_large",
                filename=file.name,
                size_mb=round(size_mb, 2),
                limit_mb=MAX_FILE_SIZE_BYTES / (1024 * 1024),
            )
            raise SecurityError(
                f"File '{file.name}' is {size_mb:.1f}MB. "
                f"Maximum allowed size is "
                f"{MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
            )

    @staticmethod
    def validate_question(question: str) -> str:
        """
        Validate user question.
        Rejects invalid input but NEVER silently mutates.
        Returns original question if valid.
        """
        if not question or not question.strip():
            raise SecurityError("Question cannot be empty.")

        if len(question) > MAX_QUESTION_LENGTH:
            raise SecurityError(
                f"Question too long ({len(question)} chars). "
                f"Maximum is {MAX_QUESTION_LENGTH} characters."
            )

        # Check for prompt injection attempts
        question_lower = question.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in question_lower:
                logger.warning(
                    "prompt_injection_attempt",
                    pattern_detected=pattern,
                    question_length=len(question),
                )
                raise SecurityError(
                    "Your question contains disallowed content. "
                    "Please ask a genuine question about your document."
                )

        # Return ORIGINAL — never mutate the user's question
        return question.strip()

    @classmethod
    def validate_file(cls, file) -> None:
        """
        Run ALL file validations in one call.
        Factory method — single entry point for file security.

        Usage:
            SecurityValidator.validate_file(uploaded_file)
        """
        cls.validate_file_type(file)
        cls.validate_file_size(file)
