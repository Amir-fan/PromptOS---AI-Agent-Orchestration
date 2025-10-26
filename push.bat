@echo off
echo Pushing to GitHub...

git add vercel.json DEPLOY_INSTRUCTIONS.md

git commit -m "Fix Vercel deployment config"

git push

echo Done!

