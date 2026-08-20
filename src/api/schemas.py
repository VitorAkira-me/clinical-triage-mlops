from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    clinical_notes: str

    @field_validator("clinical_notes")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("clinical_notes não pode ser vazio ou conter só espaços")
        return stripped


class PredictResponse(BaseModel):
    urgencia: str
    probabilidades: dict[str, float]


class HealthResponse(BaseModel):
    status: str
