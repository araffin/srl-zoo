#!/bin/bash
coverage run --source=./  -m pytest tests/
echo ""
coverage report --omit=tests/*
