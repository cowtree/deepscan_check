# deepdocs_check

- Test step by step workflow:
    - Created deepdoc_check branch
    - Add `deepdocs.yml` and create `docs/` folder
    - Add `modelcard.md` skeleton to `docs/` folder
    - Create simple ML sklearn pipeline example
    - Triggering deepdoc, merged deepdoc commit to branch
        - First modelcard update looks good: `deepdocs-update-7e7c141`
    - Updated ML pipeline with RF and HPO, model card update should take best model
        - Did not correctly chose the better model (still LR and not RF): `deepdocs-update-506ce6b`
     




    edit this to test
