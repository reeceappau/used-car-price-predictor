terraform {
  required_providers {
    render = {
      source  = "render-oss/render"
      version = "~> 1.0"
    }
  }
}

provider "render" {
  api_key  = var.render_api_key
  owner_id = var.render_owner_id
}

variable "render_api_key" {
  description = "Render API key"
  sensitive   = true
}

variable "render_owner_id" {
  description = "Render owner ID"
  sensitive   = true
}

resource "render_web_service" "bento_motors" {
  name   = "bento-motors-price-predictor"
  plan   = "free"
  region = "oregon"

  runtime_source = {
    docker = {
      repo_url    = "https://github.com/reeceappau/bento-motors-price-predictor"
      branch      = "main"
      auto_deploy = true
    }
  }
}