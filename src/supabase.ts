import { createClient } from "@supabase/supabase-js";
import type { Database } from "./types";

const supabaseURL = import.meta.env.PUBLIC_SUPABASE_URL;
const supabaseKEY = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;

export const supabase = createClient<Database>(supabaseURL, supabaseKEY);