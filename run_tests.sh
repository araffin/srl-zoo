#!/bin/bash
coverage run --source=./ --branch -m pytest tests/
echo ""
coverage report --omit=tests/*
