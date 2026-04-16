CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) NOT NULL UNIQUE,
    dietary_restrictions JSONB DEFAULT '[]',
    nutrition_goals JSONB DEFAULT '{}',
    budget_constraint FLOAT DEFAULT 100.0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recipes (
    recipe_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    ingredients JSONB NOT NULL DEFAULT '[]',
    instructions TEXT NOT NULL DEFAULT '',
    calories FLOAT,
    protein FLOAT,
    fat FLOAT,
    sodium FLOAT,
    cook_time_min INT,
    cuisine_tags JSONB DEFAULT '[]',
    source VARCHAR(50) DEFAULT 'kaggle',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pantry_items (
    pantry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    ingredient_name VARCHAR(200) NOT NULL,
    quantity FLOAT DEFAULT 1.0,
    unit VARCHAR(50) DEFAULT 'unit',
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meal_plan_interactions (
    interaction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    recipe_id UUID REFERENCES recipes(recipe_id) ON DELETE CASCADE,
    plan_week DATE NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('cooked', 'kept', 'swapped', 'skipped')),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recipe_imports (
    import_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    raw_text TEXT NOT NULL,
    model_output TEXT,
    user_correction TEXT,
    source_type VARCHAR(20) NOT NULL CHECK (source_type IN ('web_scrape', 'ocr', 'manual')),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_imports_user ON recipe_imports(user_id);
CREATE INDEX IF NOT EXISTS idx_imports_created ON recipe_imports(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON meal_plan_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON meal_plan_interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_pantry_user ON pantry_items(user_id);
