"""Application version and update source.

The release job bumps APP_VERSION to the new tag before building, so a build
always reports the release it came from. `UPDATE_REPO` is the GitHub repository
whose releases the in-app updater checks.
"""
APP_VERSION = "0.11.0"
UPDATE_REPO = "heynegix/koeiro"
