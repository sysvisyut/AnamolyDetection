from pydantic import BaseModel, ConfigDict
class Base(BaseModel):
    a: int
    model_config = ConfigDict(from_attributes=True)
class Derived(Base):
    b: int
    model_config = ConfigDict(from_attributes=True, extra="forbid")
d = Derived(a=1, b=2, c=3)
