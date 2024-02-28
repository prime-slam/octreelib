# Publish package to PyPi

1. Create tag on main branch:
    ```commandline
    git tag <tagname>
    ```
2. Push created tag to repository
    ```commandline
    git push origin <tagname>
    ```

After this GitHubActions will automatically build pip package and push it to PyPi registry.
