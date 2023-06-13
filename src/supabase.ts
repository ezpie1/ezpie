// Importing the createClient function from the @supabase/supabase-js library
import { createClient } from "@supabase/supabase-js";
// Importing the Database type from the local types file
import type { Database } from "./types";

// Storing the Supabase URL and anonymous key in constants using environment variables
const supabaseURL = import.meta.env.PUBLIC_SUPABASE_URL;
const supabaseKEY = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;

// Creating a Supabase client object with the given URL and key
export const supabase = createClient<Database>(supabaseURL, supabaseKEY);