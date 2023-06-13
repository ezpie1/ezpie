// importing useState hook from the React library
import { useState } from "react";
// importing supabase object from supabase.js
import { supabase } from "../../supabase";

// defining a function component
export default function NewsLetter() {
    // initializing name and email state variables using useState hook
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');

    // defining an async function to handle form submission
    async function handleJoin() {
        // inserting user's name and email into the NewsLetter table of the database
        const {data, error} = await supabase
        .from("NewsLetter")
        .insert({
            username: name,
            email: email,
        })
        // selecting newly inserted record's id from the database
        .select("id")
        .single()

        // displaying an alert if no data is returned
        if (!data) {
            alert("No data")
            return
        }

        // displaying an alert if there is an error while inserting data
        if (error) {
            alert("We missed up!")
            return
        }

        // displaying a success message on successful form submission
        alert("Thanks for Joining in!")
    }

    // rendering a form to collect user's name and email
    // and a button to submit the form
    return (
        <div className="text-center">
            <div className="text-white">
                <h1 className="text-3xl">Join Newsletter</h1>
                <p>Join our Newsletter to get latest blogs and sunday tips</p>
            </div>
            <div className="grid grid-cols-3 gap-4 my-5">
                <input type="text" className="p-2 rounded-lg border hover:bg-blue-600 hover:placeholder-white hover:scale-90 hover:ring hover:border-blue-400 hover:ring-blue-400 focus:outline-none focus:ring focus:bg-blue-600 focus:placeholder-white focus:border-blue-400 focus:ring-blue-400 focus:text-white duration-500 col-span-2" placeholder="Your Name" value={name} 
                onChange={e => setName(e.target.value)}/>

                <input type="text" className="p-2 rounded-lg border hover:bg-blue-600 hover:placeholder-white hover:scale-90 hover:ring hover:border-blue-400 hover:ring-blue-400 focus:outline-none focus:ring focus:bg-blue-600 focus:placeholder-white focus:border-blue-400 focus:ring-blue-400 focus:text-white duration-500" 
                placeholder="Your Email" value={email}
                onChange={e => setEmail(e.target.value)}/>
            </div>
            <button className="p-2 w-1/2 bg-white rounded-lg hover:bg-blue-600 hover:text-white hover:scale-125 duration-500" onClick={handleJoin}>Join</button>
        </div>
    )
}