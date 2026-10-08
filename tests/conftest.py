import os

# Use real Qt widgets without displaying windows or requiring audio hardware.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
