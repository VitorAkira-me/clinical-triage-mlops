from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"clinical_notes": "67yo M c/o chest pain, diaphoretic, in moderate distress"}
            ]
        }
    )

    clinical_notes: str = Field(
        ...,
        description="Texto livre do laudo/queixa clínica. Não pode ser vazio nem só espaços.",
        min_length=1,
    )

    @field_validator("clinical_notes")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("clinical_notes não pode ser vazio ou conter só espaços")
        return stripped


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "urgencia": "urgente",
                    "probabilidades": {"normal": 0.001, "atencao": 0.004, "urgente": 0.995},
                }
            ]
        }
    )

    urgencia: str = Field(
        ..., description="Classe de urgência prevista: normal, atencao ou urgente."
    )
    probabilidades: dict[str, float] = Field(
        ...,
        description="Probabilidade prevista para cada classe, nomeada pela classe (soma ~1.0).",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "ok"}]})

    status: str = Field(..., description="Confirma que o processo está no ar.")
