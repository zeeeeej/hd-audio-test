from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from src.business.retail import CatalogProvider, DialogueError, ProductFields
from src.common import error_response, request_id

router = APIRouter(prefix="/api/retail")
provider = CatalogProvider()


class ProductPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    size: str | None = None
    price_yuan: str | int | float | None = None
    stock: int | None = None
    tags: list[str] | None = None
    aliases: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_patch(cls, values):
        if not isinstance(values, dict) or not values or any(v is None for v in values.values()):
            raise ValueError("请填写要修改的商品字段，字段不能为空值")
        # Validate against the same rules as creation without inventing saved defaults.
        ProductFields.model_validate({"name": "校验", "stock": 0, "price_yuan": "0", **values})
        return values


@router.get("/catalog")
def get_catalog():
    try:
        return provider.load()
    except DialogueError as exc:
        return error_response(exc.status, exc.code, exc.message, request_id())


def save(values, product_id=None):
    try:
        product = provider.save_product(values, product_id)
        return JSONResponse(product.model_dump(), status_code=201 if product_id is None else 200)
    except ValidationError:
        return error_response(422, "invalid_product", "商品字段或数量不符合要求。", request_id())
    except DialogueError as exc:
        return error_response(exc.status, exc.code, exc.message, request_id())


@router.post("/products", status_code=201)
def create_product(product: ProductFields):
    return save(product.model_dump())


@router.patch("/products/{product_id}")
def update_product(product_id: str, product: ProductPatch):
    return save(product.model_dump(exclude_unset=True), product_id)
