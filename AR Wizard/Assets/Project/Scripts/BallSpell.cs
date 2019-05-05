using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallSpell : MonoBehaviour
{
    public ParticleSystem BallEffect, ExplosionEffect, SteamEffect;
    public Camera cam;

    private bool isBeingConjured = true;

    void Awake()
    {
        cam = GameObject.Find("ARCamera").GetComponent<Camera>();
    }

    void Start()
    {
        StartCoroutine(StartFireball());
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.tag != "Zombie") return;
        BallEffect.Stop();
        SteamEffect.Stop();
        ExplosionEffect.Play();
        GetComponent<Rigidbody>().velocity = Vector3.zero;

        StopCoroutine(StartFireball());

    }

    private IEnumerator StartFireball()
    {
        BallEffect.Stop();
        ExplosionEffect.Stop();
        SteamEffect.Play();
        yield return new WaitForSeconds(0.3f);
        BallEffect.Play();
        yield return new WaitForSeconds(2.7f);
        BallEffect.Stop();
        SteamEffect.Stop();
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        yield return null;
    }

    private void Update()
    {
        if (isBeingConjured)
        {
            //transform.position = Vector3.Lerp(transform.position, new Vector3(Hand.position.x, Hand.position.y, Hand.position.z - 1), Time.deltaTime * 50);
        }
    }

    public void StartSmoke()
    {
        SteamEffect.Play();
    }

    public void StartBall()
    {
        BallEffect.Play();
        StartCoroutine(GraduallyTurnDownEffect(2f, SteamEffect));
    }

    public void StartExplosion()
    {
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().useGravity = false;
        BallEffect.Stop();
        ExplosionEffect.Play();
    }

    public void Shoot(Vector3 direction)
    {
        //var worldDirection = cam.ViewportToWorldPoint(direction);
        isBeingConjured = false;
        GetComponent<Rigidbody>().useGravity = false;
        GetComponent<Rigidbody>().AddForce(direction.normalized * 5000);
        
    }



    private IEnumerator GraduallyTurnDownEffect(float duration, ParticleSystem ps)
    {
        float timeLeft = duration;
        float waitTime = (duration / 10);
        float emissionStepSize = ps.emission.rateOverTime.constant / 10;

        var emission = ps.emission;

        while (timeLeft > 0)
        {
            emission.rateOverTime = emission.rateOverTime.constant - emissionStepSize;
            timeLeft -= waitTime;
            yield return new WaitForSeconds(waitTime);
        }
        ps.Stop();
    }
}
