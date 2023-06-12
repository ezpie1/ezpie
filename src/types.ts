export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json }
  | Json[]

export interface Database {
  public: {
    Tables: {
      NewsLetter: {
        Row: {
          email: string | null
          id: number
          username: string | null
        }
        Insert: {
          email?: string | null
          id?: number
          username?: string | null
        }
        Update: {
          email?: string | null
          id?: number
          username?: string | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}
