#!/bin/bash
coverage run --source=./ --branch -m pytest tests/ --capture=no
echo ""
coverage report --omit=tests/*
