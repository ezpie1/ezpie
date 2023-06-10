import { useRef, useState } from "react";
import { useHover } from "usehooks-ts";
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL='https://moutdoiehpyyjejkajgc.supabase.co'
const SUPABASE_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1vdXRkb2llaHB5eWplamthamdjIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODYzMTg1OTAsImV4cCI6MjAwMTg5NDU5MH0.EEo91T-AN3V8NHYWTJzeAb6aVAG_IOnnkSNKzOMmXL4'

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY)

export default function NewsLetter() {
    const hoverRef = useRef(null);
    const isHover = useHover(hoverRef);
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');

    // Function to add a row to the newsletter table in Supabase
    const addToNewsletter = async () => {
        try {
            const {data, error} = await supabase
                .from('Newsletter')
                .insert([{
                    username: name,
                    email: email
                }]);

            if (error) {
                alert("Ops! There's been a bug! Please report at https://github.com/ezpieco/ezpie/issues")
            } else {
                alert("Thanks for joining in! You will be the first to receive latest blogs")
            }
        } catch (error) {
            console.log('Error adding row to newsletter table:', error.message)
        }
    }

    // Handle form submission
    const handleSubmit = (e) => {
        e.preventDefault();
        addToNewsletter();
    };

    return (
        <div className="text-center">
            <div className="text-white mb-5">
                <h2 className="text-2xl">Join Newsletter</h2>
                <p>
                    Join our newsletter to receive the latest blogs and Sunday tips.
                </p>
            </div>
            <div className="flex flex-col">
                <form onSubmit={handleSubmit}>
                    <div className="grid grid-cols-3 gap-3">
                        <input
                            type="text"
                            placeholder="Your Name"
                            className="my-3 rounded-md p-2 border-none hover:scale-75 bg-black dark:bg-white focus:scale-100 focus:bg-blue-600 focus:text-white duration-500"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                        />
                        <input
                            type="email"
                            placeholder="Your Email"
                            className="my-3 rounded-md p-2 border-none hover:scale-75 bg-black dark:bg-white focus:scale-100 focus:bg-blue-600 focus:text-white duration-500 col-span-2"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                    </div>
                    <button
                        type="submit"
                        className="my-3 bg-blue-400 p-2 w-1/2 mx-auto rounded-lg hover:scale-110 hover:bg-blue-600 hover:text-white duration-500"
                        ref={hoverRef}
                    >
                        {isHover ? 'Join!' : 'Waiting...'}
                    </button>
                </form>
            </div>
        </div>
    );
}
