#!/bin/bash

POSTS_REPOS=https://${GITHUB_TOKEN}@github.com/cotes2020/blog-posts.git
META_REPOS=https://${GITHUB_TOKEN}@github.com/cotes2020/blog-meta.git

GH_DEPLOY=https://${GITHUB_TOKEN}@github.com/cotes2020/cotes2020.github.io.git

POSTS_CACHE=../blog-posts
META_CACHE=../blog-meta
DEPLOY_CACHE=../deploy


init() {
  # skip if build is triggered by pull request
  if [ $TRAVIS_PULL_REQUEST == "true" ]; then
    echo "this is PR, exiting"
    exit 0
  fi

  # enable error reporting to the console
  set -e

  if [ -d "_site" ]; then
    rm -rf _site
  fi

  if [ -d  ${POSTS_CACHE} ]; then
    rm -rf ${POSTS_CACHE}
  fi

  if [ -d ${META_CACHE} ]; then
    rm -rf ${META_CACHE}
  fi
}


combine() {
  TEMPLATE=(
    "tabs/about.md"
    "LICENSE"
    "README.md"
    "_posts"
    "categories"
    "tags"
    "norobots"
    "assets/img/sample")

  for i in "${!TEMPLATE[@]}"
  do
    rm -rf ${TEMPLATE[${i}]}
  done

  git clone ${POSTS_REPOS} ${POSTS_CACHE}
  cp -a ./* ${POSTS_CACHE}
  echo "[INFO] Combined posts."

  git clone --depth=1 ${META_REPOS} ${META_CACHE}
  cp -a ${META_CACHE}/* ${POSTS_CACHE}
  rm -rf ${META_CACHE}
  echo "[INFO] Combined meta-data."
}


build() {
  cd ${POSTS_CACHE}
  python ./_scripts/tools/pages_generator.py

  # build Jekyll ouput to directory ./_site
  JEKYLL_ENV=production bundle exec jekyll build
}


deploy() {
  # Git settings
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis-CI"

  # echo "[INFO] TRAVIS_BUILD_DIR=${TRAVIS_BUILD_DIR}"
  echo
  echo "[INFO] \$PWD=$(pwd)"

  if [ -d "$DEPLOY_CACHE" ]; then
    rm -rf $DEPLOY_CACHE
  fi

  git clone --depth=1 $GH_DEPLOY $DEPLOY_CACHE

  rm -rf $DEPLOY_CACHE/*
  cp -r _site/* $DEPLOY_CACHE

  cd $DEPLOY_CACHE
  git add -A
  git commit -m "Travis-CI automated deployment #${TRAVIS_BUILD_NUMBER} of the framework."
  git push $GH_DEPLOY master:master

  echo "[INFO] Push to remote: ${GH_DEPLOY}"
}


main() {
  init
  combine
  build
  deploy
}


main
